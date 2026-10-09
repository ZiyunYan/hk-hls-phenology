#!/usr/bin/env python3
"""Sample vegetation EVI: observations, Hong Kong gap-fill, then SG9 smooth.

Ten pages, nine pixels each. The phenology run used Savitzky–Golay
(window 9, polynomial 2) on the filled EVI. The |z|>=4 spike filter is
earlier, inside the 3-day composite, so the open circles are already cleaned.
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
from phenology_hplm import STEPS, greenness
from phenology_smooth import smooth_year_stack

TILE = 256
HEIGHT = 1830
N_PAGE = 9
N_PAGES = 10


def choose_pages(seed: int, path: Path | None = None) -> list[tuple[tuple[int, int], np.ndarray]]:
    """Ten separated tiles, nine vegetation pixels each, mixed success."""
    path = PHENO / "phenology_sos_eos_hk_evi.nc" if path is None else path
    with Dataset(path) as f:
        row = np.array(f.variables["row"][:], np.int32)
        col = np.array(f.variables["col"][:], np.int32)
        sos = np.array(f.variables["sos"][:])
        eos = np.array(f.variables["eos"][:])
    nvalid = (np.isfinite(sos) & np.isfinite(eos)).sum(1).astype(np.int16)
    ty, tx = row // TILE, col // TILE
    tiles: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(zip(ty.tolist(), tx.tolist())):
        tiles.setdefault(key, []).append(i)
    ranked = sorted(tiles.items(), key=lambda kv: -len(kv[1]))
    picked: list[tuple[int, int]] = []
    for key, _idxs in ranked:
        if all(abs(key[0] - a) > 1 or abs(key[1] - b) > 1 for a, b in picked):
            picked.append(key)
        if len(picked) == N_PAGES:
            break
    rng = np.random.default_rng(seed)
    pages = []
    for key in sorted(picked):
        idxs = np.array(tiles[key], np.int32)
        keep = idxs[nvalid[idxs] >= 3]
        blank = idxs[nvalid[idxs] < 3]
        n_keep = min(5, keep.size, N_PAGE)
        n_blank = min(N_PAGE - n_keep, blank.size)
        n_keep = min(keep.size, N_PAGE - n_blank)
        take = []
        if n_keep:
            take.append(rng.choice(keep, n_keep, replace=False))
        if n_blank:
            take.append(rng.choice(blank, n_blank, replace=False))
        chosen = np.concatenate(take)
        rng.shuffle(chosen)
        pages.append((key, chosen))
    return pages, row, col, sos, eos, nvalid


def load_tile(path: Path, y0: int, x0: int) -> np.ndarray:
    y1, x1 = min(y0 + TILE, HEIGHT), min(x0 + TILE, HEIGHT)
    with Dataset(path) as f:
        return np.array(f.variables["data"][:, :, y0:y1, x0:x1], np.int16)


def page_plot(times, obs, filled, smooth, rows, cols, sos, eos, nvalid, path: Path, page: int, origin: tuple[int, int], title: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 11.2), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        ax.plot(times, filled[i], color="#9aa7b5", lw=0.7, label="Gap-filled" if i == 0 else None, zorder=2)
        ax.plot(times, smooth[i], color="#1b7f4e", lw=1.35, label="SG9 smooth" if i == 0 else None, zorder=3)
        ax.scatter(
            times, obs[i], s=7, facecolors="white", edgecolors="black", linewidths=0.35,
            label="Observed" if i == 0 else None, zorder=4,
        )
        for year in range(sos.shape[1]):
            if np.isfinite(sos[i, year]):
                ax.axvline(pd.Timestamp(year=2015 + year, month=1, day=1) + pd.Timedelta(days=float(sos[i, year]) - 1),
                           color="#1b7f4e", lw=0.45, alpha=0.55, zorder=1)
            if np.isfinite(eos[i, year]):
                ax.axvline(pd.Timestamp(year=2015 + year, month=1, day=1) + pd.Timedelta(days=float(eos[i, year]) - 1),
                           color="#b86e00", lw=0.45, alpha=0.55, zorder=1)
        finite = np.concatenate([
            filled[i][np.isfinite(filled[i])],
            obs[i][np.isfinite(obs[i])],
        ])
        lo, hi = np.percentile(finite, [1, 99])
        pad = 0.12 * max(hi - lo, 0.15)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(times[0], times[-1])
        ax.grid(True, axis="y", lw=0.3, alpha=0.45)
        ax.tick_params(labelsize=8)
        ax.set_title(f"{int(rows[i])}, {int(cols[i])}    valid {int(nvalid[i])}/11", fontsize=10, loc="left")
        if i % 3 == 0:
            ax.set_ylabel("EVI")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=10)
    fig.suptitle(
        f"Page {page}/10    tile origin row {origin[0]}, col {origin[1]}\n"
        f"{title} Gray: filled series. Green: Savitzky–Golay window 9 "
        "(~27 days), the curve used for SOS/EOS.\n"
        "Open circles: observations. Green ticks: SOS. Brown ticks: EOS. "
        "Years without ticks failed the logistic fit.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures/evi_sg9_samples")
    p.add_argument("--raw", type=Path, default=PHENO / "veg_cube.nc")
    p.add_argument("--filled", type=Path, default=PHENO / "veg_filled.nc")
    p.add_argument("--pheno", type=Path, default=PHENO / "phenology_sos_eos_hk_evi.nc")
    p.add_argument("--title", default="Hong Kong 6-band fill, EVI.")
    args = p.parse_args()
    pages, row, col, sos, eos, nvalid = choose_pages(args.seed, args.pheno)
    with Dataset(args.raw) as f:
        times = pd.to_datetime([str(t) for t in f.variables["time"][:]], format="%Y%j")
    raw_path = args.raw
    filled_path = args.filled
    for page, (key, idxs) in enumerate(pages, start=1):
        y0, x0 = key[0] * TILE, key[1] * TILE
        rows = row[idxs] - y0
        cols = col[idxs] - x0
        raw = load_tile(raw_path, y0, x0)
        filled = load_tile(filled_path, y0, x0)
        raw_px = raw[:, :, rows, cols].transpose(2, 0, 1)
        filled_px = filled[:, :, rows, cols].transpose(2, 0, 1)
        del raw, filled
        observed = (raw_px[:, :, 0] > 0) & (raw_px[:, :, 2] > 0) & (raw_px[:, :, 3] > 0)
        obs = np.where(observed, greenness(raw_px, "evi"), np.nan)
        rec = greenness(filled_px, "evi")
        sm = smooth_year_stack(rec, "sg9").reshape(rec.shape[0], -1)
        out = args.out / f"page_{page:02d}.png"
        page_plot(
            times, obs, rec, sm,
            row[idxs], col[idxs], sos[idxs], eos[idxs], nvalid[idxs],
            out, page, (y0, x0), args.title,
        )
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
