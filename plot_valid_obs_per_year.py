#!/usr/bin/env python3
"""Boxplot of valid 3-day composite counts per calendar year (2000 random veg pixels)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cube_io import load_veg_pixels
from hk_paths import PHENO
from phenology_hplm import N_YEARS, STEPS, YEAR0
from plot_style import use_arial, year_stats_table

BLUE, RED, NIR = 0, 2, 3
MAX_PER_YEAR = STEPS  # 122


def valid_mask(block: np.ndarray, rule: str) -> np.ndarray:
    """block (N, T, 6) -> (N, T) bool."""
    if rule == "all6":
        return np.all(block != 0, axis=2)
    if rule == "red_nir":
        return (block[:, :, RED] != 0) & (block[:, :, NIR] != 0)
    if rule == "evi":
        return (
            (block[:, :, BLUE] != 0)
            & (block[:, :, RED] != 0)
            & (block[:, :, NIR] != 0)
        )
    raise ValueError(rule)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument(
        "--rule",
        choices=("evi", "red_nir", "all6"),
        default="evi",
        help="valid timestep if required bands are non-zero (DN=0 missing)",
    )
    p.add_argument("--src", type=Path, default=PHENO / "veg_cube.nc")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "figures/valid_obs_per_year_n2000.png",
    )
    args = p.parse_args()
    use_arial()

    yy, xx, data = load_veg_pixels(args.n, args.seed, src=args.src)
    n_time = data.shape[1]
    if n_time != N_YEARS * STEPS:
        raise SystemExit(f"expected {N_YEARS * STEPS} steps, got {n_time}")
    ok = valid_mask(data, args.rule)
    counts = np.zeros((ok.shape[0], N_YEARS), np.int16)
    for y in range(N_YEARS):
        sl = ok[:, y * STEPS : (y + 1) * STEPS]
        counts[:, y] = sl.sum(axis=1)

    years = np.arange(YEAR0, YEAR0 + N_YEARS)
    means = counts.mean(axis=0)
    stds = counts.std(axis=0, ddof=1)

    print(f"pixels={counts.shape[0]} rule={args.rule} max_possible={MAX_PER_YEAR}/year")
    print("year  mean  std  median  p25  p75")
    for i, yr in enumerate(years):
        c = counts[:, i]
        print(
            f"{yr}  {means[i]:6.2f}  {stds[i]:5.2f}  "
            f"{np.median(c):6.1f}  {np.percentile(c, 25):5.1f}  {np.percentile(c, 75):5.1f}"
        )

    fig = plt.figure(figsize=(11, 5.8))
    ax = fig.add_axes([0.08, 0.28, 0.9, 0.62])
    bp = ax.boxplot(
        [counts[:, i] for i in range(N_YEARS)],
        tick_labels=[str(y) for y in years],
        showfliers=False,
        patch_artist=True,
        widths=0.55,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#c5dff0")
        patch.set_edgecolor("#1f4e79")
        patch.set_linewidth(0.9)
    for med in bp["medians"]:
        med.set_color("#1f4e79")
        med.set_linewidth(1.2)
    ax.axhline(
        MAX_PER_YEAR,
        color="#c0392b",
        ls="--",
        lw=1,
        label=f"Theoretical max ({MAX_PER_YEAR} bins yr$^{-1}$)",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Valid 3-day composites per pixel")
    ax.set_title(
        f"Hong Kong stable vegetation ($n$={counts.shape[0]}, seed={args.seed})\n"
        f"EVI bands DN$\\neq$0 · 3-day steps"
    )
    ax.set_ylim(0, MAX_PER_YEAR + 8)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    year_stats_table(fig, years, means, stds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
