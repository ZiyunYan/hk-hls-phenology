#!/usr/bin/env python3
"""All cleaned EVI observations for 90 vegetation pixels, 3x3 pages.

Y limits are the min and max of that pixel, so every remaining point is on
the axes. No line is drawn through gaps.
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
from phenology_hplm import greenness
from plot_evi_smooth_samples import TILE, choose_pages, load_tile


def page_plot(times, evi, rows, cols, path: Path, page: int, origin: tuple[int, int]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 11.2), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        finite = evi[i][np.isfinite(evi[i])]
        n = int(finite.size)
        if n:
            lo, hi = float(finite.min()), float(finite.max())
        else:
            lo, hi = 0.0, 1.0
        pad = 0.06 * max(hi - lo, 0.05)
        ax.set_ylim(lo - pad, hi + pad)
        ax.scatter(
            times, evi[i], s=11, facecolors="white", edgecolors="black", linewidths=0.4, zorder=3,
        )
        ax.axhline(0.0, color="#bbbbbb", lw=0.4, zorder=1)
        ax.set_xlim(times[0], times[-1])
        ax.grid(True, axis="y", lw=0.3, alpha=0.45)
        ax.tick_params(labelsize=8)
        ax.set_title(
            f"{int(rows[i])}, {int(cols[i])}    n={n}    {lo:.2f} to {hi:.2f}",
            fontsize=10, loc="left",
        )
        if i % 3 == 0:
            ax.set_ylabel("EVI")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(
        f"Page {page}/10    tile origin row {origin[0]}, col {origin[1]}\n"
        "Every EVI observation left after Bolton cleaning. Axis is the min and max of that pixel.\n"
        "No gap filling and no line through missing dates.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures/bolton_kept_points")
    args = p.parse_args()
    pages, row, col, _sos, _eos, _nvalid = choose_pages(args.seed)
    src = PHENO / "veg_cube_bolton.nc"
    with Dataset(src) as f:
        times = pd.to_datetime([str(t) for t in f.variables["time"][:]], format="%Y%j")
    spans = []
    for page, (key, idxs) in enumerate(pages, start=1):
        y0, x0 = key[0] * TILE, key[1] * TILE
        rows = row[idxs] - y0
        cols = col[idxs] - x0
        raw = load_tile(src, y0, x0)
        pixels = raw[:, :, rows, cols].transpose(2, 0, 1)
        del raw
        observed = (pixels[:, :, 0] > 0) & (pixels[:, :, 2] > 0) & (pixels[:, :, 3] > 0)
        evi = np.where(observed, greenness(pixels, "evi"), np.nan)
        for i, idx in enumerate(idxs):
            finite = evi[i][np.isfinite(evi[i])]
            spans.append((float(np.ptp(finite)) if finite.size else 0.0, page, int(row[idx]), int(col[idx]), int(finite.size)))
        out = args.out / f"page_{page:02d}.png"
        page_plot(times, evi, row[idxs], col[idxs], out, page, (y0, x0))
        print(f"wrote {out}", flush=True)
    spans.sort(reverse=True)
    print("widest EVI ranges", spans[:8], flush=True)


if __name__ == "__main__":
    main()
