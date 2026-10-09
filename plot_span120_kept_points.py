#!/usr/bin/env python3
"""Same 90 pixels after a second despike with the gap limit raised to 120 days."""
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

from bolton_clean import BLUE, RED, NIR, _gather, _neighbor_index, evi2_from_dn
from hk_paths import PHENO
from phenology_hplm import greenness
from plot_evi_smooth_samples import TILE, choose_pages, load_tile


def extra_mask(pixels: np.ndarray, day: np.ndarray, max_span: float = 120.0) -> np.ndarray:
    valid = (pixels[:, :, BLUE] > 0) & (pixels[:, :, RED] > 0) & (pixels[:, :, NIR] > 0)
    evi = np.where(valid, evi2_from_dn(pixels[:, :, RED], pixels[:, :, NIR]), np.nan)
    prev, nxt = _neighbor_index(valid)
    has = (prev >= 0) & (nxt < pixels.shape[1]) & valid
    day_grid = np.broadcast_to(day, valid.shape)
    day_prev = _gather(day_grid, prev, np.nan)
    day_next = _gather(day_grid, nxt, np.nan)
    evi_prev = _gather(np.where(valid, evi, 0.0), prev, np.nan)
    evi_next = _gather(np.where(valid, evi, 0.0), nxt, np.nan)
    span = day_next - day_prev
    weight = (day[None, :] - day_prev) / np.where(span > 0, span, np.nan)
    fitted = evi_prev * (1.0 - weight) + evi_next * weight
    return has & np.isfinite(fitted) & (span < max_span) & ((fitted - evi) > 0.1) & (evi_prev > evi) & (evi_next > evi)


def page_plot(times, kept, dropped, rows, cols, path: Path, page: int, origin: tuple[int, int]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 11.2), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        finite = kept[i][np.isfinite(kept[i])]
        n = int(finite.size)
        n_drop = int(np.isfinite(dropped[i]).sum())
        if n:
            lo, hi = float(finite.min()), float(finite.max())
        else:
            lo, hi = 0.0, 1.0
        both = np.concatenate([finite, dropped[i][np.isfinite(dropped[i])]]) if n_drop else finite
        if both.size:
            lo, hi = float(both.min()), float(both.max())
        pad = 0.06 * max(hi - lo, 0.05)
        ax.set_ylim(lo - pad, hi + pad)
        ax.scatter(times, kept[i], s=11, facecolors="white", edgecolors="black", linewidths=0.4, zorder=3, label="Kept" if i == 0 else None)
        ax.scatter(times, dropped[i], s=18, c="#c0392b", marker="x", linewidths=0.8, zorder=4, label="Removed, gap ≤ 120 d" if i == 0 else None)
        ax.axhline(0.0, color="#bbbbbb", lw=0.4, zorder=1)
        ax.set_xlim(times[0], times[-1])
        ax.grid(True, axis="y", lw=0.3, alpha=0.45)
        ax.tick_params(labelsize=8)
        ax.set_title(f"{int(rows[i])}, {int(cols[i])}    kept {n}    removed {n_drop}", fontsize=10, loc="left")
        if i % 3 == 0:
            ax.set_ylabel("EVI")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=10)
    fig.suptitle(
        f"Page {page}/10    tile origin row {origin[0]}, col {origin[1]}\n"
        "Second despike on the cleaned spectra. Red crosses would be removed: EVI2 more than 0.1\n"
        "below both neighbors, and the neighbors are at most 120 days apart. Axis includes those crosses.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures/bolton_span120_points")
    args = p.parse_args()
    pages, row, col, *_ = choose_pages(args.seed)
    src = PHENO / "veg_cube_bolton.nc"
    with Dataset(src) as f:
        times = pd.to_datetime([str(t) for t in f.variables["time"][:]], format="%Y%j")
    day = times.to_numpy().astype("datetime64[D]").astype(np.float32)
    removed = 0
    kept_n = 0
    for page, (key, idxs) in enumerate(pages, start=1):
        y0, x0 = key[0] * TILE, key[1] * TILE
        raw = load_tile(src, y0, x0)
        pixels = raw[:, :, row[idxs] - y0, col[idxs] - x0].transpose(2, 0, 1)
        del raw
        drop = extra_mask(pixels, day, 120.0)
        observed = (pixels[:, :, 0] > 0) & (pixels[:, :, 2] > 0) & (pixels[:, :, 3] > 0)
        evi = np.where(observed, greenness(pixels, "evi"), np.nan)
        kept = np.where(observed & ~drop, evi, np.nan)
        dropped = np.where(drop, evi, np.nan)
        removed += int(drop.sum())
        kept_n += int(np.isfinite(kept).sum())
        out = args.out / f"page_{page:02d}.png"
        page_plot(times, kept, dropped, row[idxs], col[idxs], out, page, (y0, x0))
        print(f"wrote {out}", flush=True)
    print(f"sample kept {kept_n} removed {removed} ({100 * removed / max(kept_n + removed, 1):.2f}%)", flush=True)


if __name__ == "__main__":
    main()
