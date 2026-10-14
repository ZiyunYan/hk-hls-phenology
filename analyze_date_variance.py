#!/usr/bin/env python3
"""Year-to-year spread of 30 m green-up and dormancy.

Pixels need six valid years. The maps are the standard deviation across years.
The right-hand panels show the land cover of the high-SD pixels and the one
environmental factor that tracks that spread, northness.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt
from scipy.stats import linregress, pearsonr

from analyze_global_extremes import load_hgt, sample_elev
from plot_style import arial, use_arial

ROOT = Path(__file__).resolve().parent
PHENO = ROOT / "data/phenology/phenology_threshold_sl122_evi.nc"
CUBE = ROOT / "data/phenology/veg_cube.nc"
DEM = ROOT / "data/dem"
FINAL = ROOT / "figures/final"
FIG = ROOT / "figures"
X0, Y0, PIX = 799980.0, 2500020.0, 30.0
HEIGHT = 1830
MIN_YEARS = 6
URBAN = (1, 2, 3, 11, 21, 22, 23, 31, 41, 42, 43, 44)
COVER = ((71, "Woodland"), (72, "Shrubland"), (73, "Grassland"), (74, "Mangrove"))


def log(msg: str) -> None:
    print(msg, flush=True)


def load_dates() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(PHENO) as src:
        row = np.array(src.variables["row"][:], np.int32)
        col = np.array(src.variables["col"][:], np.int32)
        sos = np.array(src.variables["sos15"][:], np.float32)
        eos = np.array(src.variables["eos50"][:], np.float32)
    with Dataset(CUBE) as src:
        lum = np.array(src.variables["lum_code"][:], np.int16)
    short = np.isfinite(sos) & np.isfinite(eos) & (eos <= sos + 30.0)
    sos[short] = np.nan
    eos[short] = np.nan
    return row, col, sos, eos, lum


def pixel_sd(dates: np.ndarray) -> np.ndarray:
    count = np.isfinite(dates).sum(axis=1)
    sd = np.nanstd(dates, axis=1).astype(np.float32)
    sd[count < MIN_YEARS] = np.nan
    return sd


def paint_pixels(row: np.ndarray, col: np.ndarray, values: np.ndarray) -> np.ndarray:
    image = np.full((HEIGHT, HEIGHT), np.nan, np.float32)
    ok = np.isfinite(values)
    image[row[ok], col[ok]] = values[ok]
    return image


def topography(row: np.ndarray, col: np.ndarray, lum: np.ndarray) -> dict[str, np.ndarray]:
    tiles = {113: load_hgt(DEM / "N22E113.hgt"), 114: load_hgt(DEM / "N22E114.hgt")}
    east = X0 + PIX * (np.arange(HEIGHT) + 0.5)
    north = Y0 - PIX * (np.arange(HEIGHT) + 0.5)
    ee, nn = np.meshgrid(east, north)
    to_ll = __import__("pyproj").Transformer.from_crs(32649, 4326, always_xy=True)
    lon, lat = to_ll.transform(ee, nn)
    elev = sample_elev(np.asarray(lon, np.float64), np.asarray(lat, np.float64), tiles)
    dz_row, dz_col = np.gradient(elev, PIX)
    dz_north, dz_east = -dz_row, dz_col
    horiz = np.hypot(dz_east, dz_north)
    slope = np.degrees(np.arctan(horiz)).astype(np.float32)
    northness = np.zeros(elev.shape, np.float32)
    steep = horiz >= 0.02
    northness[steep] = (-dz_north[steep] / horiz[steep]).astype(np.float32)
    sea = distance_transform_edt(lum > 0).astype(np.float32) * PIX
    urban = distance_transform_edt(~np.isin(lum, URBAN)).astype(np.float32) * PIX
    return {
        "elev": elev[row, col],
        "slope": slope[row, col],
        "northness": northness[row, col],
        "sea_m": sea[row, col],
        "urban_m": urban[row, col],
        "lum": lum[row, col],
    }


def cover_share(lum: np.ndarray, mask: np.ndarray) -> np.ndarray:
    taken = lum[mask]
    return np.array([np.mean(taken == code) for code, _name in COVER], np.float64)


def map_panel(ax, image, extent, title, norm, cmap, font) -> object:
    show = np.ma.masked_invalid(image)
    im = ax.imshow(show, origin="upper", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
    med = float(np.nanmedian(image))
    ax.set_title(f"{title}\nmedian {med:.0f} d", fontproperties=font, fontsize=12, loc="left")
    ax.set_xlabel("Longitude", fontproperties=font)
    ax.set_ylabel("Latitude", fontproperties=font)
    ax.tick_params(labelsize=8)
    return im


def scatter_panel(ax, x, y, title, font, ylim) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    slope, intercept, r, _p, _se = linregress(x[finite], y[finite])
    rng = np.random.default_rng(20261014)
    take = rng.choice(np.flatnonzero(finite), size=6000, replace=False)
    ax.scatter(x[take], y[take], s=8, c="#1f4e79", alpha=0.28, linewidths=0, rasterized=True)
    xs = np.linspace(-1.0, 1.0, 40)
    ax.plot(xs, intercept + slope * xs, color="#c0392b", lw=1.6)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0, ylim)
    ax.set_title(f"{title}\nr = {r:+.2f}", fontproperties=font, fontsize=12, loc="left")
    ax.set_xlabel("Northness (-1 south, +1 north)", fontproperties=font)
    ax.set_ylabel("SD (days)", fontproperties=font)
    ax.tick_params(labelsize=9)
    log(f"{title}  r {r:+.3f}  n {int(finite.sum())}")


def draw(sos_img, eos_img, extent, shares, sos_sd, eos_sd, north, sos_cut, eos_cut) -> None:
    use_arial()
    font = arial(11)
    small = arial(10)
    fig = plt.figure(figsize=(13.4, 8.8), layout="constrained")
    gs = fig.add_gridspec(6, 2, width_ratios=[1.15, 1.05])
    ax_a = fig.add_subplot(gs[0:3, 0])
    ax_b = fig.add_subplot(gs[3:6, 0])
    ax_c = fig.add_subplot(gs[0:2, 1])
    ax_d = fig.add_subplot(gs[2:4, 1])
    ax_e = fig.add_subplot(gs[4:6, 1])
    finite = np.concatenate([sos_img[np.isfinite(sos_img)], eos_img[np.isfinite(eos_img)]])
    vmax = float(np.ceil(np.percentile(finite, 98) / 5.0) * 5.0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")
    norm = Normalize(0, vmax)
    im = map_panel(ax_a, sos_img, extent, "(a)  Green-up (15%)", norm, cmap, font)
    map_panel(ax_b, eos_img, extent, "(b)  Dormancy (50%)", norm, cmap, font)
    cbar = fig.colorbar(im, ax=[ax_a, ax_b], fraction=0.046, pad=0.06)
    cbar.set_label("SD (days)", fontproperties=font, fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    labels = [name for _code, name in COVER]
    x = np.arange(len(labels))
    width = 0.25
    series = (
        (shares["all"], "#bdbdbd", "All pixels"),
        (shares["sos"], "#1b7f4e", f"Green-up SD ≥ {sos_cut:.0f} d"),
        (shares["eos"], "#b86e00", f"Dormancy SD ≥ {eos_cut:.0f} d"),
    )
    for i, (values, color, label) in enumerate(series):
        ax_c.bar(x + (i - 1) * width, 100 * values, width=width, color=color, label=label)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, fontproperties=small, fontsize=10)
    ax_c.set_ylabel("Share (%)", fontproperties=font)
    ax_c.set_ylim(0, 70)
    ax_c.set_title("(c)  Land cover of high-SD pixels", fontproperties=font, fontsize=12, loc="left")
    ax_c.legend(frameon=False, prop=arial(9), loc="upper right")
    ax_c.tick_params(labelsize=9)
    scatter_panel(ax_d, north, sos_sd, "(d)  Green-up and northness", font, vmax)
    scatter_panel(ax_e, north, eos_sd, "(e)  Dormancy and northness", font, vmax)
    path = FINAL / "10_phenology_interannual_sd.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"wrote {path}")


def main() -> None:
    row, col, sos, eos, lum = load_dates()
    sos_sd = pixel_sd(sos)
    eos_sd = pixel_sd(eos)
    fields = topography(row, col, lum)
    meta = json.loads((ROOT / "data/meta.json").read_text())["bounds_lonlat"]
    extent = [meta["west"], meta["east"], meta["south"], meta["north"]]
    sos_ok = np.isfinite(sos_sd)
    eos_ok = np.isfinite(eos_sd)
    sos_cut = float(np.percentile(sos_sd[sos_ok], 90))
    eos_cut = float(np.percentile(eos_sd[eos_ok], 90))
    shares = {
        "all": cover_share(fields["lum"], sos_ok),
        "sos": cover_share(fields["lum"], sos_ok & (sos_sd >= sos_cut)),
        "eos": cover_share(fields["lum"], eos_ok & (eos_sd >= eos_cut)),
    }
    for key, values in shares.items():
        text = "  ".join(f"{name} {100 * value:.1f}%" for (_code, name), value in zip(COVER, values))
        log(f"{key:4} {text}")
    rows = []
    for factor in ("elev", "slope", "northness", "sea_m", "urban_m"):
        for target, sd, ok in (("green-up", sos_sd, sos_ok), ("dormancy", eos_sd, eos_ok)):
            x = fields[factor]
            mask = ok & np.isfinite(x)
            r, p = pearsonr(x[mask], sd[mask])
            rows.append({"factor": factor, "target": target, "r": r, "p": p, "n": int(mask.sum())})
            log(f"{target:10} {factor:10} r {r:+.3f}")
    draw(
        paint_pixels(row, col, sos_sd),
        paint_pixels(row, col, eos_sd),
        extent,
        shares,
        sos_sd,
        eos_sd,
        fields["northness"],
        sos_cut,
        eos_cut,
    )
    out = FIG / "phenology_scale_variance.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
