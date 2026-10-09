#!/usr/bin/env python3
"""Compare sl122, sl244 and sl366 fills on the cleaned cube.

Observations are kept, so every model passes through them. The lines differ
in the gaps. Roughness is the mean absolute second difference of EVI.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from hk_paths import PHENO
from phenology_hplm import evi

MODELS = (122, 244, 366)
COLORS = {122: "#2166ac", 244: "#e08214", 366: "#1a9850"}
TILE = 256
HEIGHT = 1830
RAW = PHENO / "veg_cube_bolton.nc"
OUT = Path(__file__).resolve().parent / "figures" / "window_fill"


def cube(seq: int) -> Path:
    return PHENO / f"veg_filled_sl{seq}.nc"


def series_evi(block: np.ndarray) -> np.ndarray:
    """block (time, band, pixel) DN -> EVI (pixel, time). Missing DN stays NaN."""
    blue = block[:, 0, :].T.astype(np.float32)
    red = block[:, 2, :].T.astype(np.float32)
    nir = block[:, 3, :].T.astype(np.float32)
    missing = (blue <= 0) | (red <= 0) | (nir <= 0)
    out = evi(blue, red, nir)
    return np.where(missing, np.nan, out)


def roughness(y: np.ndarray) -> float:
    d2 = y[:, 2:] - 2.0 * y[:, 1:-1] + y[:, :-2]
    return float(np.mean(np.abs(d2)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with Dataset(RAW) as src:
        times = pd.to_datetime([str(t) for t in src.variables["time"][:]], format="%Y%j")
        veg = np.array(src.variables["vegetation"][:], np.uint8) > 0
    rng = np.random.default_rng(20261009)
    tiles = [
        (y0, x0)
        for y0 in range(0, HEIGHT, TILE)
        for x0 in range(0, HEIGHT, TILE)
        if np.any(veg[y0:y0 + TILE, x0:x0 + TILE])
    ]
    score_tiles = [tiles[i] for i in rng.choice(len(tiles), size=6, replace=False)]
    stats = {seq: {"fit": [], "rough": [], "n": 0} for seq in MODELS}
    plot_raw = plot_fill = None
    plot_rows = plot_cols = None

    for n_tile, (y0, x0) in enumerate(score_tiles):
        y1, x1 = min(y0 + TILE, HEIGHT), min(x0 + TILE, HEIGHT)
        local = veg[y0:y1, x0:x1]
        rows, cols = np.nonzero(local)
        with Dataset(RAW) as src:
            raw = np.array(src.variables["data"][:, :, y0:y1, x0:x1], np.int16)
        raw_pix = raw[:, :, rows, cols]
        obs = series_evi(raw_pix)
        observed = np.isfinite(obs)
        if plot_raw is None:
            take = rng.choice(rows.size, size=9, replace=False)
            plot_raw = obs[take]
            plot_rows = rows[take] + y0
            plot_cols = cols[take] + x0
            plot_fill = {}
        for seq in MODELS:
            with Dataset(cube(seq)) as src:
                filled = np.array(src.variables["data"][:, :, y0:y1, x0:x1], np.int16)
            hat = series_evi(filled[:, :, rows, cols])
            err = np.abs(hat - obs)
            stats[seq]["fit"].append(float(np.nanmean(np.where(observed, err, np.nan))))
            stats[seq]["rough"].append(roughness(np.nan_to_num(hat, nan=0.0)))
            stats[seq]["n"] += int(rows.size)
            if y0 == score_tiles[0][0] and x0 == score_tiles[0][1]:
                plot_fill[seq] = hat[take]
        print(f"tile {n_tile + 1}/6 {y0},{x0} pixels {rows.size}", flush=True)

    summary = {}
    for seq in MODELS:
        summary[str(seq)] = {
            "pixels": stats[seq]["n"],
            "evi_mae_at_observations": float(np.mean(stats[seq]["fit"])),
            "evi_roughness": float(np.mean(stats[seq]["rough"])),
        }
        print(seq, summary[str(seq)], flush=True)
    (PHENO / "window_fill_smoothness.json").write_text(json.dumps(summary, indent=2) + "\n")

    show = (times >= "2019-01-01") & (times < "2021-01-01")
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 11.2), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        for seq in MODELS:
            ax.plot(
                times[show], plot_fill[seq][i, show], color=COLORS[seq], lw=1.15,
                label=f"sl{seq}" if i == 0 else None, zorder=2,
            )
        ax.scatter(
            times[show], plot_raw[i, show], s=16, facecolors="white", edgecolors="black",
            linewidths=0.45, zorder=4, label="Observed" if i == 0 else None,
        )
        finite = plot_raw[i, show]
        finite = finite[np.isfinite(finite)]
        lo, hi = (float(np.min(finite)), float(np.max(finite))) if finite.size else (0.0, 0.6)
        pad = 0.12 * max(hi - lo, 0.1)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{int(plot_rows[i])}, {int(plot_cols[i])}", loc="left", fontsize=10)
        ax.grid(True, axis="y", lw=0.3, alpha=0.4)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        if i % 3 == 0:
            ax.set_ylabel("EVI")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=4, fontsize=10)
    fig.suptitle(
        "2019–2020 EVI. Dots are cleaned observations. Lines are the three fills.\n"
        "Observations are kept, so the lines meet the dots. Gaps show which fill is a curve and which is a polyline.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUT / "compare_2019_2020.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
