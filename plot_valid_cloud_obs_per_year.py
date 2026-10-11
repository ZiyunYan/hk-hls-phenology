#!/usr/bin/env python3
"""Valid 3-day composites (veg_cube) vs cloud-flagged granule hits (fmask) per year."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from composite_veg_cube import CLOUDISH, ROOT, STEPS_PER_YEAR, WATER_BIT, granule_jobs
from cube_io import load_veg_pixels
from hk_paths import PHENO
from phenology_hplm import N_YEARS, STEPS, YEAR0

BLUE, RED, NIR = 0, 2, 3
FMASK_FILL = 255


def valid_evi_steps(block: np.ndarray) -> np.ndarray:
    """(N, T) bool — timestep valid for EVI in veg_cube."""
    return (
        (block[:, :, BLUE] != 0)
        & (block[:, :, RED] != 0)
        & (block[:, :, NIR] != 0)
    )


def granule_cloud_clear_counts(yy: np.ndarray, xx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel per-year counts of granule observations flagged cloudy vs clear (fmask)."""
    n_pix = yy.size
    cloud = np.zeros((n_pix, N_YEARS), np.int32)
    clear = np.zeros((n_pix, N_YEARS), np.int32)
    jobs = granule_jobs()
    for i, (gid, _sensor, bin_index) in enumerate(jobs):
        year_idx = bin_index // STEPS_PER_YEAR
        path = ROOT / "granules" / f"{gid}.npz"
        with np.load(path) as z:
            fv = z["fmask"][yy, xx]
        is_cloud = (fv == FMASK_FILL) | ((fv.astype(np.uint8) & CLOUDISH) != 0)
        is_clear = (fv != FMASK_FILL) & ((fv.astype(np.uint8) & CLOUDISH) == 0) & (
            (fv.astype(np.uint8) & WATER_BIT) == 0
        )
        cloud[:, year_idx] += is_cloud.astype(np.int32)
        clear[:, year_idx] += is_clear.astype(np.int32)
        if (i + 1) % 200 == 0 or i + 1 == len(jobs):
            print(f"granules {i + 1}/{len(jobs)}", flush=True)
    return cloud, clear


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--src", type=Path, default=PHENO / "veg_cube.nc")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "figures/valid_cloud_obs_per_year_n2000.png",
    )
    args = p.parse_args()

    yy, xx, data = load_veg_pixels(args.n, args.seed, src=args.src)
    ok = valid_evi_steps(data)
    valid = np.zeros((ok.shape[0], N_YEARS), np.int16)
    for y in range(N_YEARS):
        valid[:, y] = ok[:, y * STEPS : (y + 1) * STEPS].sum(axis=1)

    print("scanning granule fmask …", flush=True)
    cloud, clear = granule_cloud_clear_counts(yy, xx)

    years = np.arange(YEAR0, YEAR0 + N_YEARS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, counts, title, color, ymax_hint in [
        (
            axes[0],
            valid,
            "Valid 3-day composites (veg_cube, EVI bands DN≠0)",
            "#c5dff0",
            STEPS,
        ),
        (
            axes[1],
            cloud,
            "Cloud-flagged granule observations (HLS fmask: cloud/snow/shadow/cirrus/fill)",
            "#f5c5c5",
            None,
        ),
    ]:
        bp = ax.boxplot(
            [counts[:, i] for i in range(N_YEARS)],
            tick_labels=[str(y) for y in years],
            showfliers=False,
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor("#333")
        if ymax_hint:
            ax.axhline(ymax_hint, color="#c0392b", ls="--", lw=1, label=f"max bins/yr ({ymax_hint})")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_xlabel("Year")
        ax.set_ylabel("Count per pixel")
        ax.set_title(title)
        means = counts.mean(axis=0)
        stds = counts.std(axis=0, ddof=1)
        for i, yr in enumerate(years):
            ax.text(
                i + 1,
                ax.get_ylim()[1] * 0.98 if ymax_hint else counts[:, i].max() + 1,
                f"μ={means[i]:.0f}\nσ={stds[i]:.1f}",
                ha="center",
                va="top",
                fontsize=7,
                color="#333",
            )

    fig.suptitle(
        f"Hong Kong stable vegetation n={valid.shape[0]} seed={args.seed}\n"
        f"Right: one count per granule overpass where fmask is cloudy; left: successful composite timesteps",
        fontsize=10,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    print("\nvalid composite — year mean std | cloud granule — year mean std")
    for i, yr in enumerate(years):
        print(
            f"{yr}  {valid[:, i].mean():6.2f} {valid[:, i].std(ddof=1):5.2f}  |  "
            f"{cloud[:, i].mean():6.2f} {cloud[:, i].std(ddof=1):5.2f}  "
            f"(clear granule μ={clear[:, i].mean():.1f})"
        )


if __name__ == "__main__":
    main()
