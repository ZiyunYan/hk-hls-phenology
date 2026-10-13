#!/usr/bin/env python3
"""Median green-up and dormancy maps for SG windows, fill weights, and a cubic spline."""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from netCDF4 import Dataset
from numba import njit, prange
from scipy.signal import savgol_filter

from phenology_hplm import N_YEARS, STEPS, YEAR0, greenness
from phenology_threshold import threshold_years
from plot_style import arial, use_arial

ROOT = Path(__file__).resolve().parent
FILLED = ROOT / "data/phenology/veg_filled_sl122.nc"
BOLTON = ROOT / "data/phenology/veg_cube_bolton.nc"
FIG = ROOT / "figures"
HEIGHT = 1830
MIN_YEARS = 3
FILL_WEIGHT = 0.25
WEST, SOUTH, EAST, NORTH = 113.90703502249974, 22.074877118149995, 114.45054741024268, 22.58053002866876


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def second_difference(n: int) -> np.ndarray:
    d = np.zeros((n - 2, n), np.float64)
    idx = np.arange(n - 2)
    d[idx, idx] = 1.0
    d[idx, idx + 1] = -2.0
    d[idx, idx + 2] = 1.0
    return d


def kernel_width_steps(row: np.ndarray) -> int:
    center = row.size // 2
    mass = np.abs(row)
    total = mass.sum()
    half = 0
    covered = mass[center]
    while covered < 0.5 * total and half < center:
        half += 1
        covered += mass[center - half] + mass[center + half]
    return 2 * half + 1


def spline_hat(n: int, target_width: int) -> tuple[np.ndarray, float, int]:
    """Penalized cubic smoothing spline on a regular grid. Width is the central kernel span."""
    dtd = second_difference(n).T @ second_difference(n)
    eye = np.eye(n)
    best = None
    lam = 1.0
    for _ in range(28):
        hat = np.linalg.inv(eye + lam * dtd)
        width = kernel_width_steps(hat[n // 2])
        err = abs(width - target_width)
        if best is None or err < best[0]:
            best = (err, lam, width, hat)
        lam *= 1.35
    assert best is not None
    return best[3], best[1], best[2]


@njit
def _solve_center(a00, a01, a02, a11, a12, a22, b0, b1, b2):
    a00 += 1e-6
    a11 += 1e-6
    a22 += 1e-6
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    det = a00 * c00 + a01 * c01 + a02 * c02
    if abs(det) < 1e-14:
        return np.nan
    return (c00 * b0 + c01 * b1 + c02 * b2) / det


@njit(parallel=True)
def weighted_sg(y, w, window):
    """Local quadratic regression. y, w are (pixels, time). Returns the fitted value."""
    n, t = y.shape
    half = window // 2
    out = np.empty((n, t), np.float64)
    for i in prange(n):
        for c in range(t):
            if c < half:
                lo = 0
            elif c >= t - half:
                lo = t - window
            else:
                lo = c - half
            hi = lo + window
            a00 = a01 = a02 = a11 = a12 = a22 = 0.0
            b0 = b1 = b2 = 0.0
            for k in range(lo, hi):
                ww = w[i, k]
                if ww <= 0.0:
                    continue
                rel = float(k - c)
                yy = y[i, k]
                rel2 = rel * rel
                a00 += ww
                a01 += ww * rel
                a02 += ww * rel2
                a11 += ww * rel2
                a12 += ww * rel2 * rel
                a22 += ww * rel2 * rel2
                b0 += ww * yy
                b1 += ww * rel * yy
                b2 += ww * rel2 * yy
            out[i, c] = _solve_center(a00, a01, a02, a11, a12, a22, b0, b1, b2)
    return out


def equal_sg(years: np.ndarray, window: int) -> np.ndarray:
    n, _, t = years.shape
    flat = years.reshape(-1, t).astype(np.float64, copy=True)
    bad = ~np.isfinite(flat)
    if bad.any():
        med = np.nanmedian(np.where(bad, np.nan, flat), axis=1)
        med = np.where(np.isfinite(med), med, 0.0)
        flat = np.where(bad, med[:, None], flat)
    sm = savgol_filter(flat, window_length=window, polyorder=2, axis=-1, mode="interp")
    sm[bad] = np.nan
    return sm.reshape(n, N_YEARS, t).astype(np.float32)


def weighted_years(years: np.ndarray, observed: np.ndarray, window: int) -> np.ndarray:
    n, ny, t = years.shape
    out = np.empty((n, ny, t), np.float32)
    for year in range(ny):
        y = years[:, year].astype(np.float64, copy=True)
        finite = np.isfinite(y)
        w = np.where(observed[:, year] & finite, 1.0, np.where(finite, FILL_WEIGHT, 0.0))
        if not finite.all():
            med = np.nanmedian(np.where(finite, y, np.nan), axis=1)
            med = np.where(np.isfinite(med), med, 0.0)
            y = np.where(finite, y, med[:, None])
        sm = weighted_sg(y, w, window).astype(np.float32)
        sm[~finite] = np.nan
        out[:, year] = sm
    return out


def spline_years(years: np.ndarray, hat: np.ndarray) -> np.ndarray:
    n, ny, t = years.shape
    flat = years.reshape(-1, t).astype(np.float64, copy=True)
    bad = ~np.isfinite(flat)
    if bad.any():
        med = np.nanmedian(np.where(bad, np.nan, flat), axis=1)
        med = np.where(np.isfinite(med), med, 0.0)
        flat = np.where(bad, med[:, None], flat)
    sm = flat @ hat.T
    sm[bad] = np.nan
    return sm.reshape(n, ny, t).astype(np.float32)


def dates_of(smooth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sos15, _, eos50, _ = threshold_years(np.ascontiguousarray(smooth))
    return sos15, eos50


def median_map(row, col, values) -> np.ndarray:
    grid = np.full((HEIGHT, HEIGHT), np.nan, np.float32)
    nfin = np.isfinite(values).sum(axis=1)
    keep = nfin >= MIN_YEARS
    if np.any(keep):
        grid[row[keep], col[keep]] = np.nanmedian(values[keep], axis=1)
    return grid


def draw(columns: list[tuple[str, np.ndarray, np.ndarray]], path: Path, suptitle: str) -> None:
    use_arial()
    font = arial(11)
    letter = arial(12)
    sos_stack = np.concatenate([col[1][np.isfinite(col[1])] for col in columns])
    eos_stack = np.concatenate([col[2][np.isfinite(col[2])] for col in columns])
    sos_lim = np.nanpercentile(sos_stack, [2, 98])
    eos_lim = np.nanpercentile(eos_stack, [2, 98])
    fig_w = 3.15 * len(columns) + 0.7
    fig, axes = plt.subplots(2, len(columns), figsize=(fig_w, 7.15), dpi=140, squeeze=False)
    fig.subplots_adjust(left=0.045, right=0.92, top=0.88, bottom=0.06, wspace=0.08, hspace=0.16)
    extent = [WEST, EAST, SOUTH, NORTH]
    letters = "abcdefghijklmnopqrstuvwxyz"
    images = []
    for col, (title, sos, eos) in enumerate(columns):
        for row, (grid, cmap, lim, ylabel) in enumerate((
            (sos, "viridis", sos_lim, "Green-up (day)"),
            (eos, "cividis", eos_lim, "Dormancy (day)"),
        )):
            ax = axes[row, col]
            show = np.ma.masked_invalid(grid)
            palette = plt.get_cmap(cmap).copy()
            palette.set_bad("#d0d0d0")
            im = ax.imshow(
                show, origin="upper", extent=extent, cmap=palette,
                norm=Normalize(lim[0], lim[1]), interpolation="nearest", aspect="equal",
            )
            images.append((im, row, ylabel))
            ax.set_title(title, fontproperties=font, color="black", pad=3)
            if row == 1:
                ax.set_xlabel("Longitude", fontproperties=font, color="black")
            else:
                ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel("Latitude", fontproperties=font, color="black")
                ax.text(
                    0.0, 1.08, f"({letters[row]})",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontproperties=letter, color="black", clip_on=False,
                )
            else:
                ax.set_ylabel("")
            ax.tick_params(labelsize=8, colors="black")
            if col > 0:
                ax.set_yticklabels([])
    for row, ylabel in ((0, "Green-up (day)"), (1, "Dormancy (day)")):
        im = next(image for image, r, _ in images if r == row)
        cax = fig.add_axes([0.935, 0.55 if row == 0 else 0.12, 0.012, 0.30])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(ylabel, fontproperties=font, color="black")
        cbar.ax.tick_params(labelsize=8, colors="black")
    fig.suptitle(suptitle, fontproperties=arial(13), color="black", y=0.98)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"wrote {path}")


def main() -> None:
    hat, lam, width = spline_hat(STEPS, target_width=15)
    log(f"cubic spline lambda {lam:.2f}, central kernel {width} steps ({3 * (width - 1)} days span)")
    # Warm the compiler and check the weighted fit against a dense normal equation.
    y = np.linspace(0.2, 0.6, STEPS)[None, :] + 0.0
    w = np.ones_like(y)
    w[:, ::3] = FILL_WEIGHT
    got = weighted_sg(y, w, 9)[0, 40]
    log(f"weighted SG compile check center {got:.4f}")

    with Dataset(FILLED) as src:
        veg = np.array(src.variables["vegetation"][:], np.uint8) > 0
    n = int(veg.sum())
    log(f"vegetation pixels {n}")
    methods = ("sg15", "sg11", "sg9", "sg7", "sg15w", "sg9w", "sg7w", "spline")
    sos = {name: np.full((n, N_YEARS), np.nan, np.float32) for name in methods}
    eos = {name: np.full((n, N_YEARS), np.nan, np.float32) for name in methods}
    rows_all = np.empty(n, np.int32)
    cols_all = np.empty(n, np.int32)
    cursor = 0
    t0 = time.time()
    strip = 48
    with Dataset(FILLED) as filled, Dataset(BOLTON) as bolton:
        fd, bd = filled.variables["data"], bolton.variables["data"]
        for y0 in range(0, HEIGHT, strip):
            y1 = min(y0 + strip, HEIGHT)
            local = veg[y0:y1]
            rr, cc = np.nonzero(local)
            if rr.size == 0:
                continue
            raw_f = np.array(fd[:, :4, y0:y1, :], np.int16)
            raw_b = np.array(bd[:, :4, y0:y1, :], np.int16)
            block_f = raw_f[:, :, rr, cc].transpose(2, 0, 1)
            block_b = raw_b[:, :, rr, cc].transpose(2, 0, 1)
            del raw_f, raw_b
            series = greenness(block_f, "evi").reshape(-1, N_YEARS, STEPS)
            seen = (
                (block_b[:, :, 0] != 0) & (block_b[:, :, 2] != 0) & (block_b[:, :, 3] != 0)
            ).reshape(-1, N_YEARS, STEPS)
            del block_f, block_b
            smooth = {
                "sg15": equal_sg(series, 15),
                "sg11": equal_sg(series, 11),
                "sg9": equal_sg(series, 9),
                "sg7": equal_sg(series, 7),
                "sg15w": weighted_years(series, seen, 15),
                "sg9w": weighted_years(series, seen, 9),
                "sg7w": weighted_years(series, seen, 7),
                "spline": spline_years(series, hat),
            }
            nstrip = rr.size
            for name, sm in smooth.items():
                a, c = dates_of(sm)
                sos[name][cursor : cursor + nstrip] = a
                eos[name][cursor : cursor + nstrip] = c
            rows_all[cursor : cursor + nstrip] = rr + y0
            cols_all[cursor : cursor + nstrip] = cc
            cursor += nstrip
            log(f"rows {y1}/{HEIGHT} pixels {cursor}/{n} ({time.time() - t0:.0f}s)")
    if cursor != n:
        raise RuntimeError(f"pixel count {cursor} != {n}")

    maps = {}
    for name in methods:
        maps[name] = (
            median_map(rows_all, cols_all, sos[name]),
            median_map(rows_all, cols_all, eos[name]),
        )
        s, e = maps[name]
        log(
            f"{name:8} green-up median {np.nanmedian(s):6.1f}  std {np.nanstd(s):5.1f}"
            f"   dormancy median {np.nanmedian(e):6.1f}  std {np.nanstd(e):5.1f}"
            f"   mapped {(np.isfinite(s).sum())}"
        )
    base_s, base_e = maps["sg15"]
    for name in methods:
        if name == "sg15":
            continue
        s, e = maps[name]
        both_s = np.isfinite(s) & np.isfinite(base_s)
        both_e = np.isfinite(e) & np.isfinite(base_e)
        log(
            f"{name:8} minus SG15  green-up {np.nanmedian(s[both_s] - base_s[both_s]):+.1f} d"
            f"   dormancy {np.nanmedian(e[both_e] - base_e[both_e]):+.1f} d"
        )

    equal_cols = [
        ("SG 15, equal weight", *maps["sg15"]),
        ("SG 11, equal weight", *maps["sg11"]),
        ("SG 9, equal weight", *maps["sg9"]),
        ("SG 7, equal weight", *maps["sg7"]),
        (f"Cubic spline, {3 * (width - 1)} days", *maps["spline"]),
    ]
    weight_cols = [
        ("SG 15, equal weight", *maps["sg15"]),
        ("SG 15, fill 0.25", *maps["sg15w"]),
        ("SG 9, fill 0.25", *maps["sg9w"]),
        ("SG 7, fill 0.25", *maps["sg7w"]),
    ]
    year = f"{YEAR0}–{YEAR0 + N_YEARS - 1} median"
    final = FIG / "final"
    draw(equal_cols, final / "sos_eos_sg_and_spline.png", f"Green-up and dormancy, {year}")
    draw(weight_cols, final / "sos_eos_fill_weight.png", f"Green-up and dormancy, {year}")

    # One year, so the salt-and-pepper that a MODIS cell average sees is visible.
    year_ix = 2020 - YEAR0
    def one_year(values):
        grid = np.full((HEIGHT, HEIGHT), np.nan, np.float32)
        ok = np.isfinite(values[:, year_ix])
        grid[rows_all[ok], cols_all[ok]] = values[ok, year_ix]
        return grid

    year_cols = [
        ("SG 15, equal weight", one_year(sos["sg15"]), one_year(eos["sg15"])),
        ("SG 9, fill 0.25", one_year(sos["sg9w"]), one_year(eos["sg9w"])),
        (f"Cubic spline, {3 * (width - 1)} days", one_year(sos["spline"]), one_year(eos["spline"])),
    ]
    draw(year_cols, FIG / "final" / "sos_eos_2020_three_smooth.png", "Green-up and dormancy, 2020")


if __name__ == "__main__":
    main()
