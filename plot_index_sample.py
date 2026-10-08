#!/usr/bin/env python3
"""Plot NDVI, EVI and EVI2 for 100 random vegetation pixels.

Reflectance is HLS DN / 10000, clipped to [0, 1], matching phenology_hplm.evi2.
Bands are Blue, Green, Red, NIR (L30 B05 / S30 B8A), SWIR1, SWIR2.

NDVI = (NIR - Red) / (NIR + Red)
EVI  = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
EVI2 = 2.5 * (NIR - Red) / (NIR + 2.4*Red + 1)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from hk_paths import PHENO

SRC = PHENO / "veg_cube.nc"
DST = PHENO / "veg_filled.nc"
BLUE, RED, NIR = 0, 2, 3
TILE = 256


def reflectance(dn: np.ndarray) -> np.ndarray:
    return np.clip(dn.astype(np.float32) / 10000.0, 0.0, 1.0)


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = nir + red
    out = np.full(red.shape, np.nan, np.float32)
    ok = denom > 1e-6
    out[ok] = (nir[ok] - red[ok]) / denom[ok]
    return out


def evi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    out = np.full(red.shape, np.nan, np.float32)
    ok = np.abs(denom) > 1e-6
    out[ok] = 2.5 * (nir[ok] - red[ok]) / denom[ok]
    return out


def evi2(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = nir + 2.4 * red + 1.0
    out = np.full(red.shape, np.nan, np.float32)
    ok = np.abs(denom) > 1e-6
    out[ok] = 2.5 * (nir[ok] - red[ok]) / denom[ok]
    return out


def sample_pixels(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(SRC) as src:
        veg = np.array(src.variables["vegetation"][:], np.uint8) > 0
        times = pd.to_datetime([str(t) for t in src.variables["time"][:]], format="%Y%j")
    ys, xs = np.nonzero(veg)
    take = np.random.default_rng(seed).choice(ys.size, size=n, replace=False)
    return ys[take].astype(np.int32), xs[take].astype(np.int32), times


def _tile_job(job: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y0, x0, y1, x1, rows, cols, idxs = job
    with Dataset(SRC) as src, Dataset(DST) as dst:
        raw = np.array(src.variables["data"][:, :, y0:y1, x0:x1], dtype=np.int16)
        filled = np.array(dst.variables["data"][:, :, y0:y1, x0:x1], dtype=np.int16)
    return (
        idxs,
        raw[:, :, rows, cols].transpose(2, 0, 1),
        filled[:, :, rows, cols].transpose(2, 0, 1),
    )


def load_series(ys: np.ndarray, xs: np.ndarray, workers: int) -> tuple[np.ndarray, np.ndarray]:
    import multiprocessing as mp

    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (y, x) in enumerate(zip(ys, xs)):
        buckets.setdefault((int(y) // TILE * TILE, int(x) // TILE * TILE), []).append(i)
    jobs = []
    for (y0, x0), idxs in buckets.items():
        y1, x1 = min(y0 + TILE, 1830), min(x0 + TILE, 1830)
        rows = ys[idxs] - y0
        cols = xs[idxs] - x0
        jobs.append((y0, x0, y1, x1, rows.astype(np.int32), cols.astype(np.int32), np.array(idxs, np.int32)))
    ctx = mp.get_context("spawn")
    raw = np.empty((ys.size, 1342, 6), np.int16)
    filled = np.empty_like(raw)
    with ctx.Pool(min(workers, len(jobs))) as pool:
        for idxs, raw_i, filled_i in pool.imap_unordered(_tile_job, jobs):
            raw[idxs] = raw_i
            filled[idxs] = filled_i
    return raw, filled


def indices(cube: np.ndarray, observed: np.ndarray | None) -> dict[str, np.ndarray]:
    """cube (N, T, 6) DN. If observed is set, drop steps whose inputs are missing."""
    blue = reflectance(cube[:, :, BLUE])
    red = reflectance(cube[:, :, RED])
    nir = reflectance(cube[:, :, NIR])
    out = {"NDVI": ndvi(red, nir), "EVI": evi(blue, red, nir), "EVI2": evi2(red, nir)}
    if observed is None:
        return out
    need = {
        "NDVI": observed[:, :, RED] & observed[:, :, NIR],
        "EVI": observed[:, :, BLUE] & observed[:, :, RED] & observed[:, :, NIR],
        "EVI2": observed[:, :, RED] & observed[:, :, NIR],
    }
    for name, mask in need.items():
        out[name] = np.where(mask, out[name], np.nan)
    return out


def plot(times: pd.DatetimeIndex, obs: dict[str, np.ndarray], rec: dict[str, np.ndarray], ys, xs, path: Path) -> None:
    names = ("NDVI", "EVI", "EVI2")
    colors = {"NDVI": "#1b7f4e", "EVI": "#1f4e79", "EVI2": "#e07a1f"}
    n = ys.size
    ncol = 10
    nrow = int(np.ceil(n / ncol))
    fig_h = 3 * nrow * 1.15 + 1.4
    fig, axes = plt.subplots(3 * nrow, ncol, figsize=(22, fig_h), sharex=True, squeeze=False)
    for block, name in enumerate(names):
        finite = rec[name][np.isfinite(rec[name])]
        lo, hi = np.nanpercentile(finite, [1, 99])
        pad = 0.08 * max(hi - lo, 0.2)
        ylim = (lo - pad, hi + pad)
        for i in range(n):
            r, c = divmod(i, ncol)
            ax = axes[block * nrow + r, c]
            ax.plot(times, rec[name][i], color=colors[name], lw=1.15, zorder=2)
            ax.scatter(
                times, obs[name][i], s=9, facecolors="white", edgecolors="black",
                linewidths=0.35, zorder=3,
            )
            ax.set_ylim(*ylim)
            ax.set_xlim(times[0], times[-1])
            ax.tick_params(labelsize=6, length=2)
            ax.grid(True, axis="y", lw=0.3, alpha=0.4)
            if r == 0:
                ax.set_title(name if c == ncol // 2 else "", fontsize=11, color=colors[name], pad=2)
            if c == 0 and r == nrow // 2:
                ax.set_ylabel(name, fontsize=8, color=colors[name])
            ax.text(0.02, 0.92, f"{int(ys[i])},{int(xs[i])}", transform=ax.transAxes, fontsize=6, va="top")
            if not (block == 2 and r == nrow - 1):
                ax.tick_params(labelbottom=False)
            else:
                ax.xaxis.set_major_locator(mdates.YearLocator(2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for i in range(n, nrow * ncol):
            r, c = divmod(i, ncol)
            axes[block * nrow + r, c].axis("off")
    fig.suptitle(
        "100 random vegetation pixels, HLS 49QHE\n"
        "Line: gap-filled reconstruction    Open circles: original observations\n"
        "NDVI=(NIR−Red)/(NIR+Red)    "
        "EVI=2.5(NIR−Red)/(NIR+6Red−7.5Blue+1)    "
        "EVI2=2.5(NIR−Red)/(NIR+2.4Red+1)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures/sample100_ndvi_evi_evi2.png")
    args = p.parse_args()
    ys, xs, times = sample_pixels(args.n, args.seed)
    raw, filled = load_series(ys, xs, args.workers)
    observed = raw > 0
    obs = indices(raw, observed)
    rec = indices(filled, None)
    plot(times, obs, rec, ys, xs, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
