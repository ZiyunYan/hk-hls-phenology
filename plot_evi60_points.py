#!/usr/bin/env python3
"""Preview a 60-day EVI despike on the same 90 pixels.

Red crosses sit more than 0.1 EVI above both neighbors. Orange crosses sit
more than 0.1 below. The neighbor gap is under 60 days. The cube is unchanged.
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

from bolton_clean import _gather, _neighbor_index
from hk_paths import PHENO
from phenology_hplm import greenness
from plot_evi_smooth_samples import TILE, choose_pages, load_tile

SPAN = 60.0
RISE = 0.1


def evi_flags(pixels: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = (pixels[:, :, 0] > 0) & (pixels[:, :, 2] > 0) & (pixels[:, :, 3] > 0)
    evi = np.where(valid, greenness(pixels, "evi"), np.nan)
    prev, nxt = _neighbor_index(valid)
    has = (prev >= 0) & (nxt < pixels.shape[1]) & valid
    grid = np.broadcast_to(day, valid.shape)
    d0, d1 = _gather(grid, prev, np.nan), _gather(grid, nxt, np.nan)
    v0 = _gather(np.where(valid, evi, 0.0), prev, np.nan)
    v1 = _gather(np.where(valid, evi, 0.0), nxt, np.nan)
    span = d1 - d0
    weight = (day[None, :] - d0) / np.where(span > 0, span, np.nan)
    fitted = v0 * (1.0 - weight) + v1 * weight
    base = has & np.isfinite(fitted) & (span > 0) & (span < SPAN)
    up = base & ((evi - fitted) > RISE) & (evi > v0) & (evi > v1)
    down = base & ((fitted - evi) > RISE) & (v0 > evi) & (v1 > evi)
    return up, down


def page_plot(times, kept, up, down, rows, cols, path: Path, page: int, origin: tuple[int, int]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 11.2), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        pieces = [kept[i], up[i], down[i]]
        finite = np.concatenate([p[np.isfinite(p)] for p in pieces])
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        pad = 0.06 * max(hi - lo, 0.05)
        ax.set_ylim(lo - pad, hi + pad)
        ax.scatter(times, kept[i], s=11, facecolors="white", edgecolors="black", linewidths=0.4, zorder=3, label="Kept" if i == 0 else None)
        ax.scatter(times, up[i], s=22, c="#c0392b", marker="x", linewidths=0.9, zorder=4, label="High, gap < 60 d" if i == 0 else None)
        ax.scatter(times, down[i], s=22, c="#e67e22", marker="x", linewidths=0.9, zorder=4, label="Low, gap < 60 d" if i == 0 else None)
        ax.set_xlim(times[0], times[-1])
        ax.grid(True, axis="y", lw=0.3, alpha=0.45)
        ax.tick_params(labelsize=8)
        n_up = int(np.isfinite(up[i]).sum())
        n_down = int(np.isfinite(down[i]).sum())
        ax.set_title(
            f"{int(rows[i])}, {int(cols[i])}    kept {int(np.isfinite(kept[i]).sum())}    high {n_up}    low {n_down}",
            fontsize=10, loc="left",
        )
        if i % 3 == 0:
            ax.set_ylabel("EVI")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=10)
    fig.suptitle(
        f"Page {page}/10    tile origin row {origin[0]}, col {origin[1]}\n"
        "60-day EVI screen on the already cleaned spectra. Red: more than 0.1 above both neighbors.\n"
        "Orange: more than 0.1 below. Axis includes every point. The cube is not changed.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures/evi60_points")
    args = p.parse_args()
    pages, row, col, *_ = choose_pages(args.seed)
    src = PHENO / "veg_cube_bolton.nc"
    with Dataset(src) as f:
        times = pd.to_datetime([str(t) for t in f.variables["time"][:]], format="%Y%j")
    day = times.to_numpy().astype("datetime64[D]").astype(np.float32)
    n_up = n_down = n_kept = 0
    for page, (key, idxs) in enumerate(pages, start=1):
        y0, x0 = key[0] * TILE, key[1] * TILE
        raw = load_tile(src, y0, x0)
        pixels = raw[:, :, row[idxs] - y0, col[idxs] - x0].transpose(2, 0, 1)
        del raw
        up_m, down_m = evi_flags(pixels, day)
        observed = (pixels[:, :, 0] > 0) & (pixels[:, :, 2] > 0) & (pixels[:, :, 3] > 0)
        evi = np.where(observed, greenness(pixels, "evi"), np.nan)
        kept = np.where(observed & ~up_m & ~down_m, evi, np.nan)
        up = np.where(up_m, evi, np.nan)
        down = np.where(down_m & ~up_m, evi, np.nan)
        n_up += int(up_m.sum())
        n_down += int((down_m & ~up_m).sum())
        n_kept += int(np.isfinite(kept).sum())
        out = args.out / f"page_{page:02d}.png"
        page_plot(times, kept, up, down, row[idxs], col[idxs], out, page, (y0, x0))
        print(f"wrote {out}", flush=True)
    print(f"sample kept {n_kept} high {n_up} low {n_down}", flush=True)


if __name__ == "__main__":
    main()
