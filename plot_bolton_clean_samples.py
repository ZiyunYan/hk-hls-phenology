#!/usr/bin/env python3
"""Show Bolton et al. (2020) outlier removal on the same 90 vegetation pixels.

Open circles stay. Red crosses are bright anomalies. Orange crosses are
negative EVI2 spikes. Nothing is interpolated or gap-filled.
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

from bolton_clean import apply_bolton_clean
from hk_paths import PHENO
from phenology_hplm import greenness
from plot_evi_smooth_samples import TILE, choose_pages, load_tile

HEIGHT = 1830


def page_plot(times, kept, bright, spike, rows, cols, nvalid, path: Path, page: int, origin: tuple[int, int]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 11.2), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        pieces = [kept[i][np.isfinite(kept[i])], bright[i][np.isfinite(bright[i])], spike[i][np.isfinite(spike[i])]]
        finite = np.concatenate(pieces) if any(p.size for p in pieces) else np.array([0.0, 1.0])
        lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
        pad = 0.12 * max(hi - lo, 0.15)
        ax.set_ylim(lo - pad, hi + pad)
        ax.plot(times, kept[i], color="#1b7f4e", lw=0.9, zorder=2)
        ax.scatter(
            times, kept[i], s=8, facecolors="white", edgecolors="black", linewidths=0.35,
            label="Kept" if i == 0 else None, zorder=3,
        )
        ax.scatter(
            times, bright[i], s=16, c="#c0392b", marker="x", linewidths=0.8,
            label="Bright anomaly" if i == 0 else None, zorder=4,
        )
        ax.scatter(
            times, spike[i], s=16, c="#e67e22", marker="x", linewidths=0.8,
            label="EVI2 spike" if i == 0 else None, zorder=4,
        )
        ax.set_xlim(times[0], times[-1])
        ax.grid(True, axis="y", lw=0.3, alpha=0.45)
        ax.tick_params(labelsize=8)
        n_drop = int(np.isfinite(bright[i]).sum() + np.isfinite(spike[i]).sum())
        ax.set_title(
            f"{int(rows[i])}, {int(cols[i])}    valid {int(nvalid[i])}/11    removed {n_drop}",
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
        "Bolton et al. (2020) cleaning on the 3-day composite, before gap filling.\n"
        "Green: observations kept. Red: blue-reflectance bright anomaly. "
        "Orange: EVI2 more than 0.1 below both neighbors.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures/bolton_clean_samples")
    args = p.parse_args()
    pages, row, col, _sos, _eos, nvalid = choose_pages(args.seed)
    with Dataset(PHENO / "veg_cube.nc") as f:
        times = pd.to_datetime([str(t) for t in f.variables["time"][:]], format="%Y%j")
    day = times.to_numpy().astype("datetime64[D]").astype(np.float32)
    raw_path = PHENO / "veg_cube.nc"
    totals = {"observed": 0, "bright": 0, "spike": 0, "removed": 0}
    for page, (key, idxs) in enumerate(pages, start=1):
        y0, x0 = key[0] * TILE, key[1] * TILE
        rows = row[idxs] - y0
        cols = col[idxs] - x0
        raw = load_tile(raw_path, y0, x0)
        pixels = raw[:, :, rows, cols].transpose(2, 0, 1)
        del raw
        cleaned, stats = apply_bolton_clean(pixels, day)
        for name in totals:
            totals[name] += stats[name]
        from bolton_clean import bolton_outlier_mask
        observed = (pixels[:, :, 0] > 0) & (pixels[:, :, 2] > 0) & (pixels[:, :, 3] > 0)
        bright_mask, spike_mask = bolton_outlier_mask(pixels, day)
        evi = greenness(pixels, "evi")
        kept = np.where(observed & ~bright_mask & ~spike_mask, evi, np.nan)
        bright = np.where(bright_mask, evi, np.nan)
        spike = np.where(spike_mask & ~bright_mask, evi, np.nan)
        out = args.out / f"page_{page:02d}.png"
        page_plot(times, kept, bright, spike, row[idxs], col[idxs], nvalid[idxs], out, page, (y0, x0))
        print(f"wrote {out} {stats}", flush=True)
    removed = totals["removed"]
    observed = max(totals["observed"], 1)
    print(
        f"sample observed {totals['observed']} bright {totals['bright']} "
        f"spike {totals['spike']} removed {removed} ({100 * removed / observed:.2f}%)",
        flush=True,
    )


if __name__ == "__main__":
    main()
