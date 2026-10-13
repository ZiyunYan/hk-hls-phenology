#!/usr/bin/env python3
"""City maps, one ridge, and the scatters for the 30 m green-up contrast."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from netCDF4 import Dataset
from pyproj import Transformer
from scipy.stats import spearmanr

from plot_aspect_scatters import paired_faces, style_ax, subsample, trend, year_median
from plot_style import arial, use_arial

ROOT = Path(__file__).resolve().parent
PHENO = ROOT / "data/phenology"
FIG = ROOT / "figures/aspect_space.png"
HEIGHT = WIDTH = 1830
SLOPE_MIN = 10.0
# 3.6 km window in the central New Territories. Both faces are present,
# and the early EVI stripes follow the north-facing slopes.
RIDGE = (660, 930, 120)
DATE_LIM = (30, 140)


def grid_of(row, col, values) -> np.ndarray:
    out = np.full((HEIGHT, WIDTH), np.nan, np.float32)
    out[row, col] = values
    return out


def transformer() -> Transformer:
    crs = json.loads((ROOT / "data/meta.json").read_text())["crs"]
    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def extent_of(tr: Transformer, r0: int, c0: int, r1: int, c1: int) -> list[float]:
    x0, x1 = 799980.0 + c0 * 30.0, 799980.0 + c1 * 30.0
    y_north, y_south = 2500020.0 - r0 * 30.0, 2500020.0 - r1 * 30.0
    west, south = tr.transform(x0, y_south)
    east, north = tr.transform(x1, y_north)
    return [west, east, south, north]


def draw_map(fig, ax, grid, extent, cmap, vmin, vmax, title, cbar_label, font, small, box=None):
    show = np.ma.masked_invalid(grid)
    colors = plt.get_cmap(cmap).copy()
    colors.set_bad("#e6e6e6")
    image = ax.imshow(
        show, origin="upper", extent=extent, cmap=colors,
        norm=Normalize(vmin, vmax), interpolation="nearest", aspect="equal",
    )
    if box is not None:
        width, height = box[1] - box[0], box[3] - box[2]
        ax.add_patch(Rectangle(
            (box[0], box[2]), width, height,
            fill=False, edgecolor="black", linewidth=2.2, zorder=5,
        ))
        ax.add_patch(Rectangle(
            (box[0], box[2]), width, height,
            fill=False, edgecolor="#ffe14a", linewidth=1.0, zorder=6,
        ))
    ax.set_title(title, fontproperties=font, loc="left", color="black")
    ax.set_xlabel("Longitude", fontproperties=small)
    ax.set_ylabel("Latitude", fontproperties=small)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    decimals = "%.3f" if (extent[1] - extent[0]) < 0.1 else "%.2f"
    ax.xaxis.set_major_formatter(FormatStrFormatter(decimals))
    ax.yaxis.set_major_formatter(FormatStrFormatter(decimals))
    ax.tick_params(labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(small)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label(cbar_label, fontproperties=small)
    bar.ax.tick_params(labelsize=8)
    return bar


def northness_bar(bar, small) -> None:
    bar.set_ticks([-1, 0, 1])
    bar.set_ticklabels(["South", "Flat", "North"])
    for label in bar.ax.get_yticklabels():
        label.set_fontproperties(small)


def main() -> None:
    rng = np.random.default_rng(0)
    z = np.load(PHENO / "global_extreme_pixels.npz")
    b = np.load(PHENO / "aspect_mechanism_pixels.npz")
    row, col = z["row"], z["col"]
    sos15 = z["sos_med"]
    north = z["northness"]
    slope = z["slope"]
    ndvi = b["sos15_n"]
    winter = b["winter_cosi"]
    with Dataset(PHENO / "phenology_threshold_sl122_evi.nc") as ds:
        sos50 = year_median(np.array(ds.variables["sos50"][:], np.float32))

    steep = (slope >= SLOPE_MIN) & np.isfinite(winter) & np.isfinite(sos15) & np.isfinite(north)
    rho, _ = spearmanr(winter[steep], sos15[steep])
    faces = paired_faces(row, col, slope, north, {"sos15": sos15, "ndvi": ndvi, "sos50": sos50})
    d_evi = faces["sos15_south"] - faces["sos15_north"]
    d_ndvi = faces["ndvi_south"] - faces["ndvi_north"]
    ok = np.isfinite(d_evi) & np.isfinite(d_ndvi)
    print(f"illumination Spearman {rho:.3f} on {int(steep.sum())} steep pixels")
    print(
        f"cells {int(ok.sum())}  EVI south-north {np.median(d_evi[ok]):.1f} days"
        f"  NDVI {np.median(d_ndvi[ok]):.1f} days"
    )

    tr = transformer()
    full = extent_of(tr, 0, 0, HEIGHT, WIDTH)
    r0, c0, size = RIDGE
    zoom = extent_of(tr, r0, c0, r0 + size, c0 + size)
    date = grid_of(row, col, sos15)
    aspect = grid_of(row, col, north)

    use_arial()
    font = arial(11)
    small = arial(8)
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.8), dpi=160)

    draw_map(
        fig, axes[0, 0], date, full, "viridis", *DATE_LIM,
        "(a)  EVI 15% date", "Day of year", font, small, box=zoom,
    )
    bar = draw_map(
        fig, axes[0, 1], aspect, full, "RdBu", -1, 1,
        "(b)  Northness", "Northness", font, small, box=zoom,
    )
    northness_bar(bar, small)

    ax = axes[0, 2]
    cmap = plt.get_cmap("RdBu")
    norm = Normalize(vmin=-1, vmax=1)
    idx = subsample(steep, 50000, rng)
    sc = ax.scatter(
        winter[idx], sos15[idx], c=north[idx], cmap=cmap, norm=norm,
        s=6, alpha=0.35, linewidths=0, rasterized=True,
    )
    trend(ax, winter[steep], sos15[steep])
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(15, 160)
    ax.set_xlabel("January illumination, cos i", fontproperties=font)
    ax.set_ylabel("EVI 15% date, day of year", fontproperties=font)
    ax.set_title("(c)  Winter illumination", fontproperties=font, loc="left", color="black")
    ax.text(
        0.04, 0.96, f"Steep slopes\nSpearman {rho:.2f}",
        transform=ax.transAxes, ha="left", va="top", fontproperties=small, color="#222222",
    )
    style_ax(ax, small)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Northness", fontproperties=small)
    northness_bar(cbar, small)

    draw_map(
        fig, axes[1, 0], date[r0:r0 + size, c0:c0 + size], zoom, "viridis", *DATE_LIM,
        "(d)  Same ridge, EVI", "Day of year", font, small,
    )
    bar = draw_map(
        fig, axes[1, 1], aspect[r0:r0 + size, c0:c0 + size], zoom, "RdBu", -1, 1,
        "(e)  Same ridge, northness", "Northness", font, small,
    )
    northness_bar(bar, small)

    ax = axes[1, 2]
    lo, hi = 20, 180
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=0.9, zorder=1)
    for key, color, label in (
        ("sos15", "#2c7bb6", "EVI 15%"),
        ("ndvi", "#d7191c", "NDVI 15%"),
    ):
        x, y = faces[f"{key}_north"], faces[f"{key}_south"]
        keep = np.isfinite(x) & np.isfinite(y)
        ax.scatter(
            x[keep], y[keep], s=9, c=color, alpha=0.35, linewidths=0,
            label=label, rasterized=True,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("North face, day of year", fontproperties=font)
    ax.set_ylabel("South face, day of year", fontproperties=font)
    ax.set_title("(f)  Opposite faces, 510 m", fontproperties=font, loc="left", color="black")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(prop=small, frameon=False, loc="upper left")
    ax.text(
        0.97, 0.05,
        f"South face later\nEVI {np.median(d_evi[ok]):+.0f} days\nNDVI {np.median(d_ndvi[ok]):+.0f} days",
        transform=ax.transAxes, ha="right", va="bottom", fontproperties=small, color="#222222",
    )
    style_ax(ax, small)

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
