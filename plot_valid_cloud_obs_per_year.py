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
from plot_style import use_arial, year_stats_table

BLUE, RED, NIR = 0, 2, 3
FMASK_FILL = 255


def valid_evi_steps(block: np.ndarray) -> np.ndarray:
    return (
        (block[:, :, BLUE] != 0)
        & (block[:, :, RED] != 0)
        & (block[:, :, NIR] != 0)
    )


def granule_cloud_clear_counts(yy: np.ndarray, xx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_pix = yy.size
    cloud = np.zeros((n_pix, N_YEARS), np.int32)
    clear = np.zeros((n_pix, N_YEARS), np.int32)
    jobs = granule_jobs()
    for i, (gid, _sensor, bin_index) in enumerate(jobs):
        year_idx = bin_index // STEPS_PER_YEAR
        with np.load(ROOT / "granules" / f"{gid}.npz") as z:
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


def draw_boxpanel(
    ax,
    counts: np.ndarray,
    years: np.ndarray,
    title: str,
    facecolor: str,
    ymax_line: int | None = None,
) -> None:
    bp = ax.boxplot(
        [counts[:, i] for i in range(N_YEARS)],
        tick_labels=[str(y) for y in years],
        showfliers=False,
        patch_artist=True,
        widths=0.55,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(facecolor)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.9)
    if ymax_line is not None:
        ax.axhline(
            ymax_line,
            color="#c0392b",
            ls="--",
            lw=1,
            label=f"Max bins yr$^{-1}$ ({ymax_line})",
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.set_title(title, pad=8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Count per pixel")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--src", type=Path, default=PHENO / "veg_cube.nc")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "figures/valid_cloud_obs_per_year_n2000.png",
    )
    args = p.parse_args()
    use_arial()

    cache = args.out.parent / f".cache_fmask_counts_n{args.n}_s{args.seed}.npz"
    yy, xx, data = load_veg_pixels(args.n, args.seed, src=args.src)
    ok = valid_evi_steps(data)
    valid = np.zeros((ok.shape[0], N_YEARS), np.int16)
    for y in range(N_YEARS):
        valid[:, y] = ok[:, y * STEPS : (y + 1) * STEPS].sum(axis=1)

    if cache.exists() and not args.refresh_cache:
        z = np.load(cache)
        cloud, clear = z["cloud"], z["clear"]
        print(f"loaded cache {cache}", flush=True)
    else:
        print("scanning granule fmask …", flush=True)
        cloud, clear = granule_cloud_clear_counts(yy, xx)
        np.savez_compressed(cache, cloud=cloud, clear=clear, yy=yy, xx=xx)
        print(f"saved cache {cache}", flush=True)

    years = np.arange(YEAR0, YEAR0 + N_YEARS)
    v_mean, v_std = valid.mean(axis=0), valid.std(axis=0, ddof=1)
    c_mean, c_std = cloud.mean(axis=0), cloud.std(axis=0, ddof=1)

    fig = plt.figure(figsize=(12, 7.2))
    ax0 = fig.add_axes([0.07, 0.36, 0.42, 0.52])
    ax1 = fig.add_axes([0.55, 0.36, 0.42, 0.52])
    draw_boxpanel(
        ax0,
        valid,
        years,
        "Valid 3-day composites\n(veg_cube, EVI DN$\\neq$0)",
        "#c5dff0",
        STEPS,
    )
    draw_boxpanel(
        ax1,
        cloud,
        years,
        "Cloud-flagged granule hits\n(HLS fmask: cloud / snow / shadow / cirrus / fill)",
        "#f5c5c5",
        None,
    )
    ymax = max(cloud.max(), valid.max()) * 1.08
    ax1.set_ylim(0, ymax)
    fig.suptitle(
        f"Hong Kong stable vegetation ($n$={valid.shape[0]}, seed={args.seed})",
        fontsize=12,
        y=0.96,
    )

    year_stats_table(
        fig,
        years,
        v_mean,
        v_std,
        rect=(0.06, 0.17, 0.9, 0.13),
        extra_rows=[("Cloud mean", c_mean), ("Cloud std", c_std)],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}")

    print("\nvalid — year mean std | cloud granule — year mean std")
    for i, yr in enumerate(years):
        print(
            f"{yr}  {valid[:, i].mean():6.2f} {valid[:, i].std(ddof=1):5.2f}  |  "
            f"{cloud[:, i].mean():6.2f} {cloud[:, i].std(ddof=1):5.2f}  "
            f"(clear granule mean={clear[:, i].mean():.1f})"
        )


if __name__ == "__main__":
    main()
