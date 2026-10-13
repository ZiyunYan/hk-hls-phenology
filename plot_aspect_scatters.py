#!/usr/bin/env python3
"""Scatter plots for the 30 m green-up contrasts in Hong Kong."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from netCDF4 import Dataset
from scipy.stats import spearmanr

from plot_style import arial, use_arial

ROOT = Path(__file__).resolve().parent
PHENO = ROOT / "data/phenology"
FIG = ROOT / "figures/aspect_scatters.png"
SLOPE_MIN = 10.0
ASPECT_MIN = 0.5
CELL = 17  # 510 m
MIN_N = 8


def year_median(values: np.ndarray, min_years: int = 5) -> np.ndarray:
    count = np.isfinite(values).sum(1)
    out = np.full(values.shape[0], np.nan, np.float32)
    ok = count >= min_years
    with np.errstate(all="ignore"):
        out[ok] = np.nanmedian(values[ok], axis=1)
    return out


def paired_faces(row, col, slope, northness, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Median on the north face and the south face of each 510 m cell."""
    north = (slope >= SLOPE_MIN) & (northness >= ASPECT_MIN)
    south = (slope >= SLOPE_MIN) & (northness <= -ASPECT_MIN)
    key = (row // CELL).astype(np.int64) * 10_000 + (col // CELL).astype(np.int64)
    north_key, south_key = key[north], key[south]
    keys = np.intersect1d(np.unique(north_key), np.unique(south_key))
    north_order = np.argsort(north_key, kind="mergesort")
    south_order = np.argsort(south_key, kind="mergesort")
    north_sorted, south_sorted = north_key[north_order], south_key[south_order]
    north_idx = np.flatnonzero(north)[north_order]
    south_idx = np.flatnonzero(south)[south_order]
    n_bounds = np.flatnonzero(np.diff(north_sorted)) + 1
    s_bounds = np.flatnonzero(np.diff(south_sorted)) + 1
    n_starts, n_ends = np.r_[0, n_bounds], np.r_[n_bounds, north_sorted.size]
    s_starts, s_ends = np.r_[0, s_bounds], np.r_[s_bounds, south_sorted.size]
    n_pos = {int(north_sorted[s]): (s, e) for s, e in zip(n_starts, n_ends)}
    s_pos = {int(south_sorted[s]): (s, e) for s, e in zip(s_starts, s_ends)}
    out = {f"{name}_{side}": [] for name in fields for side in ("north", "south")}
    for kid in keys:
        a0, a1 = n_pos[int(kid)]
        b0, b1 = s_pos[int(kid)]
        if a1 - a0 < MIN_N or b1 - b0 < MIN_N:
            continue
        ni, si = north_idx[a0:a1], south_idx[b0:b1]
        kept = False
        record = {}
        for name, values in fields.items():
            nv, sv = values[ni], values[si]
            nv, sv = nv[np.isfinite(nv)], sv[np.isfinite(sv)]
            if nv.size < MIN_N or sv.size < MIN_N:
                record[name] = (np.nan, np.nan)
                continue
            record[name] = (float(np.median(nv)), float(np.median(sv)))
            kept = True
        if not kept:
            continue
        for name, (nv, sv) in record.items():
            out[f"{name}_north"].append(nv)
            out[f"{name}_south"].append(sv)
    return {name: np.asarray(vals, np.float64) for name, vals in out.items()}


def subsample(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size <= n:
        return idx
    return rng.choice(idx, n, replace=False)


def trend(ax, x, y, color="#222222") -> None:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return
    coef = np.polyfit(x[m], y[m], 1)
    xs = np.linspace(np.percentile(x[m], 1), np.percentile(x[m], 99), 50)
    ax.plot(xs, coef[0] * xs + coef[1], color=color, lw=1.4, zorder=3)


def style_ax(ax, font) -> None:
    ax.tick_params(labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)


def main() -> None:
    rng = np.random.default_rng(0)
    z = np.load(PHENO / "global_extreme_pixels.npz")
    b = np.load(PHENO / "aspect_mechanism_pixels.npz")
    sos15 = z["sos_med"]
    jan = z["jan_med"]
    north = z["northness"]
    slope = z["slope"]
    with Dataset(PHENO / "phenology_threshold_sl122_evi.nc") as ds:
        sos50 = year_median(np.array(ds.variables["sos50"][:], np.float32))
    ndvi = b["sos15_n"]
    winter = b["winter_cosi"]

    steep = slope >= SLOPE_MIN
    early = np.isfinite(sos15) & (sos15 <= np.nanpercentile(sos15, 5))
    early_ndvi = ndvi[early & np.isfinite(ndvi)]
    print(f"early EVI p5 cutoff {np.nanpercentile(sos15, 5):.1f}")
    print(f"early pixels n={int(early.sum())} northness median {np.nanmedian(north[early]):.2f}")
    print(f"early pixels NDVI 15% median {np.nanmedian(early_ndvi):.1f} p5 {np.nanpercentile(early_ndvi, 5):.1f}")
    print(f"all NDVI 15% p5 {np.nanpercentile(ndvi, 5):.1f} median {np.nanmedian(ndvi):.1f}")
    r_w, _ = spearmanr(winter[steep & np.isfinite(sos15)], sos15[steep & np.isfinite(sos15)])
    print(f"steep winter cos i vs sos15 Spearman {r_w:.3f}")
    gap = sos50 - sos15
    print(f"15-to-50 gap early median {np.nanmedian(gap[early]):.1f} all {np.nanmedian(gap):.1f}")

    faces = paired_faces(
        z["row"], z["col"], slope, north,
        {"sos15": sos15, "sos50": sos50, "ndvi": ndvi, "winter": winter},
    )
    d15 = faces["sos15_south"] - faces["sos15_north"]
    ok = np.isfinite(d15)
    print(f"cells {int(ok.sum())} median south-north 15% {np.median(d15[ok]):.1f} frac {np.mean(d15[ok] > 0):.3f}")

    use_arial()
    font = arial(11)
    small = arial(9)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.4), dpi=160)
    cmap = plt.get_cmap("RdBu")
    norm = Normalize(vmin=-1, vmax=1)

    ax = axes[0, 0]
    m = steep & np.isfinite(winter) & np.isfinite(sos15) & np.isfinite(north)
    idx = subsample(m, 50000, rng)
    sc = ax.scatter(
        winter[idx], sos15[idx], c=north[idx], cmap=cmap, norm=norm,
        s=6, alpha=0.35, linewidths=0, rasterized=True,
    )
    trend(ax, winter[m], sos15[m])
    ax.set_xlabel("January illumination, cos i", fontproperties=font)
    ax.set_ylabel("EVI 15% date, day of year", fontproperties=font)
    ax.set_title("(a)  Winter illumination", fontproperties=font, loc="left", color="black")
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(15, 160)
    style_ax(ax, small)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Northness", fontproperties=small)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["South", "Flat", "North"])
    cbar.ax.tick_params(labelsize=8)
    ax.text(0.04, 0.96, "Steep slopes\nSpearman 0.76", transform=ax.transAxes,
            ha="left", va="top", fontproperties=small, color="#222222")

    ax = axes[0, 1]
    lo, hi = 20, 180
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=0.9, zorder=1)
    for key, color, label in (
        ("sos15", "#2c7bb6", "15% date"),
        ("sos50", "#d7191c", "50% date"),
    ):
        x, y = faces[f"{key}_north"], faces[f"{key}_south"]
        keep = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[keep], y[keep], s=10, c=color, alpha=0.35, linewidths=0, label=label, rasterized=True)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("North face of the same 510 m cell", fontproperties=font)
    ax.set_ylabel("South face of the same cell", fontproperties=font)
    ax.set_title("(b)  Opposite faces of one ridge", fontproperties=font, loc="left", color="black")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(prop=small, frameon=False, loc="upper left")
    ax.text(0.97, 0.06, "Above the line:\nsouth face later", transform=ax.transAxes,
            ha="right", va="bottom", fontproperties=small, color="#222222")
    style_ax(ax, small)

    ax = axes[1, 0]
    m = np.isfinite(ndvi) & np.isfinite(sos15) & np.isfinite(north)
    idx = subsample(m, 50000, rng)
    ax.plot([20, 170], [20, 170], color="#888888", lw=0.9, zorder=1)
    sc = ax.scatter(
        ndvi[idx], sos15[idx], c=north[idx], cmap=cmap, norm=norm,
        s=6, alpha=0.35, linewidths=0, rasterized=True,
    )
    ax.set_xlim(20, 170)
    ax.set_ylim(20, 170)
    ax.set_xlabel("NDVI 15% date, day of year", fontproperties=font)
    ax.set_ylabel("EVI 15% date, day of year", fontproperties=font)
    ax.set_title("(c)  The early tail is an EVI date", fontproperties=font, loc="left", color="black")
    ax.set_aspect("equal", adjustable="box")
    style_ax(ax, small)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Northness", fontproperties=small)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["South", "Flat", "North"])
    cbar.ax.tick_params(labelsize=8)
    ax.text(0.04, 0.96, "Below the line:\nEVI earlier than NDVI", transform=ax.transAxes,
            ha="left", va="top", fontproperties=small, color="#222222")

    ax = axes[1, 1]
    m = np.isfinite(jan) & np.isfinite(sos15) & np.isfinite(north)
    idx = subsample(m, 50000, rng)
    sc = ax.scatter(
        jan[idx], sos15[idx], c=north[idx], cmap=cmap, norm=norm,
        s=6, alpha=0.35, linewidths=0, rasterized=True,
    )
    trend(ax, jan[m], sos15[m])
    ax.set_xlabel("January EVI", fontproperties=font)
    ax.set_ylabel("EVI 15% date, day of year", fontproperties=font)
    ax.set_title("(d)  Dark January, early date", fontproperties=font, loc="left", color="black")
    ax.set_xlim(0.05, 0.75)
    ax.set_ylim(15, 160)
    style_ax(ax, small)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Northness", fontproperties=small)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["South", "Flat", "North"])
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
