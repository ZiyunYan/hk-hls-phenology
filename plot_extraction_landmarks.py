#!/usr/bin/env python3
"""Median smoothed EVI with the dates used for SOS and EOS.

The curve is the median of valid seasons from the 122-day fill after
Savitzky–Golay (window 9). Markers are the median day of each landmark
across those seasons. SOS and EOS are the Zhang curvature extrema.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.signal import savgol_filter

from phenology_hplm import greenness, metrics_smoothed_years

SRC = Path(__file__).resolve().parent / "data/phenology/veg_filled_sl122.nc"
OUT = Path(__file__).resolve().parent / "figures/extraction_landmarks_sl122_evi.png"
TILE = 128
N_TILES = 6
PER_TILE = 500
DOY = 1.0 + 3.0 * np.arange(122)


def sg9(evi: np.ndarray) -> np.ndarray:
    years = np.ascontiguousarray(evi, np.float32).reshape(-1, 11, 122)
    flat = years.reshape(-1, 122).astype(np.float64, copy=True)
    bad = ~np.isfinite(flat)
    if bad.any():
        with np.errstate(all="ignore"):
            row_med = np.nanmedian(flat, axis=1)
        row_med = np.where(np.isfinite(row_med), row_med, 0.0)
        flat = np.where(bad, row_med[:, None], flat)
    sm = savgol_filter(flat, window_length=9, polyorder=2, axis=-1, mode="interp")
    sm[bad] = np.nan
    return sm.reshape(years.shape).astype(np.float32)


def choose_tiles(veg: np.ndarray) -> list[tuple[int, int]]:
    scores = []
    for y0 in range(0, veg.shape[0] - TILE + 1, TILE):
        for x0 in range(0, veg.shape[1] - TILE + 1, TILE):
            n = int(veg[y0 : y0 + TILE, x0 : x0 + TILE].sum())
            if n >= 800:
                scores.append((n, y0, x0))
    scores.sort(reverse=True)
    picked: list[tuple[int, int]] = []
    for _n, y0, x0 in scores:
        if all(abs(y0 - a) >= 2 * TILE or abs(x0 - b) >= 2 * TILE for a, b in picked):
            picked.append((y0, x0))
        if len(picked) == N_TILES:
            break
    return picked


def load_sample(path: Path) -> np.ndarray:
    rng = np.random.default_rng(0)
    blocks = []
    with Dataset(path) as f:
        veg = np.array(f.variables["vegetation"][:]) > 0
        tiles = choose_tiles(veg)
        print("tiles", tiles, flush=True)
        for y0, x0 in tiles:
            local = veg[y0 : y0 + TILE, x0 : x0 + TILE]
            rows, cols = np.nonzero(local)
            take = rng.choice(rows.size, min(PER_TILE, rows.size), replace=False)
            raw = np.array(f.variables["data"][:, :, y0 : y0 + TILE, x0 : x0 + TILE], np.int16)
            pix = raw[:, :, rows[take], cols[take]].transpose(2, 0, 1)
            blocks.append(pix)
            print(y0, x0, pix.shape[0], flush=True)
            del raw
    return np.concatenate(blocks, 0)


def first_cross(y: np.ndarray, i0: int, i1: int, level: float, rising: bool) -> float:
    seg = y[i0 : i1 + 1]
    hit = np.flatnonzero(seg >= level) if rising else np.flatnonzero(seg <= level)
    if hit.size == 0:
        return np.nan
    return float(DOY[i0 + int(hit[0])])


def landmarks(sm: np.ndarray, sos: np.ndarray, eos: np.ndarray) -> dict[str, np.ndarray]:
    ok = np.isfinite(sos) & np.isfinite(eos)
    idx = np.argwhere(ok)
    acc = {k: [] for k in ("left", "p15", "sos", "p50", "peak", "d50", "eos", "d15", "right")}
    for i, year in idx:
        y = sm[i, year]
        finite = np.isfinite(y)
        if int(finite.sum()) < 12:
            continue
        p = int(np.nanargmax(np.where(finite, y, -1e9)))
        if p < 4 or p > 117:
            continue
        left = int(np.nanargmin(np.where(finite[: p + 1], y[: p + 1], 1e9)))
        right = p + int(np.nanargmin(np.where(finite[p:], y[p:], 1e9)))
        amp = float(y[p] - y[left])
        amp_d = float(y[p] - y[right])
        if amp < 0.04 or amp_d < 0.04:
            continue
        acc["left"].append(float(DOY[left]))
        acc["peak"].append(float(DOY[p]))
        acc["right"].append(float(DOY[right]))
        acc["sos"].append(float(sos[i, year]))
        acc["eos"].append(float(eos[i, year]))
        acc["p15"].append(first_cross(y, left, p, y[left] + 0.15 * amp, True))
        acc["p50"].append(first_cross(y, left, p, y[left] + 0.50 * amp, True))
        acc["d50"].append(first_cross(y, p, right, y[p] - 0.50 * amp_d, False))
        acc["d15"].append(first_cross(y, p, right, y[right] + 0.15 * amp_d, False))
    return {k: np.asarray(v, np.float64) for k, v in acc.items()}


def median_curve(sm: np.ndarray, sos: np.ndarray, eos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ok = np.isfinite(sos) & np.isfinite(eos)
    curves = sm[ok]
    finite = np.isfinite(curves)
    curves = np.where(finite, curves, np.nan)
    med = np.nanmedian(curves, axis=0)
    lo = np.nanpercentile(curves, 25, axis=0)
    hi = np.nanpercentile(curves, 75, axis=0)
    return med, lo, hi


def y_at(curve: np.ndarray, day: float) -> float:
    return float(np.interp(day, DOY, curve))


def main() -> None:
    pix = load_sample(SRC)
    print("pixels", pix.shape[0], flush=True)
    sm = sg9(greenness(pix, "evi"))
    sos, eos = metrics_smoothed_years(np.ascontiguousarray(sm))
    marks = landmarks(sm, sos, eos)
    med, lo, hi = median_curve(sm, sos, eos)
    n = marks["sos"].size
    stats = {}
    for key, vals in marks.items():
        q = np.nanpercentile(vals, [25, 50, 75])
        stats[key] = q
        print(f"{key:6} n={np.isfinite(vals).sum():5d}  p25={q[0]:6.1f}  p50={q[1]:6.1f}  p75={q[2]:6.1f}")
    print("right on last step", float(np.mean(marks["right"] >= 361)))

    series = [
        ("spring minimum", "left", "#5c6b73", "o"),
        ("15% of rise", "p15", "#2f6f9f", "s"),
        ("SOS, curvature", "sos", "#1b7f4e", "^"),
        ("50% of rise", "p50", "#2f6f9f", "D"),
        ("peak", "peak", "#1a1a1a", "o"),
        ("50% of decline", "d50", "#b86e00", "D"),
        ("EOS, curvature", "eos", "#b86e00", "^"),
        ("15% above floor", "d15", "#8a4b08", "s"),
        ("year-end minimum", "right", "#5c6b73", "x"),
    ]

    crest = float(DOY[int(np.nanargmax(med))])
    print("median-curve crest", crest, flush=True)

    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.fill_between(DOY, lo, hi, color="#1b7f4e", alpha=0.16, lw=0, label="Middle half of seasons", zorder=1)
    ax.plot(DOY, med, color="#1b7f4e", lw=2.0, label="Median smoothed EVI", zorder=2)
    ax.axvline(83, color="#6b4c9a", ls="--", lw=0.9, zorder=0)
    ax.axvline(308, color="#6b4c9a", ls="--", lw=0.9, zorder=0)
    ax.plot([], [], color="#6b4c9a", ls="--", lw=0.9, label="Greater Bay Area MODIS, day 83 and 308")
    ax.scatter(
        [crest], [float(np.nanmax(med))], s=42, facecolors="none", edgecolors="#1a1a1a",
        lw=1.2, zorder=5, label=f"Crest of this median curve, day {crest:.0f}",
    )

    ymin = float(np.nanmin(lo))
    ymax = float(np.nanmax(hi))
    pad = 0.08 * (ymax - ymin)
    rug = ymin - 0.55 * pad
    ax.set_ylim(rug - 0.35 * pad, ymax + pad)

    for label, key, color, marker in series:
        day = float(stats[key][1])
        lo_d, hi_d = float(stats[key][0]), float(stats[key][2])
        y = y_at(med, day)
        ax.plot([lo_d, hi_d], [rug, rug], color=color, lw=2.4, alpha=0.95, zorder=3, solid_capstyle="butt")
        ax.plot([day, day], [rug, y], color=color, lw=0.6, alpha=0.55, zorder=3)
        ax.scatter([day], [y], s=42, color=color, marker=marker, zorder=4, label=f"{label}, day {day:.0f}")

    ax.set_xlim(1, 366)
    ax.set_xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("EVI")
    ax.set_xlabel("Day of year")
    ax.grid(True, axis="y", lw=0.3, alpha=0.45)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8.5)
    ax.set_title(
        f"Where the dates sit on the smoothed EVI  ·  {n:,} valid seasons, 122-day fill, SG window 9\n"
        "Dots are median dates on the median curve. Bottom bars are the middle half of each date.",
        loc="left",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
