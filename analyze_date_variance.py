#!/usr/bin/env python3
"""Year-to-year spread of 30 m green-up and dormancy.

Pixels need six valid years. The maps are the standard deviation across years.
The right-hand panel is the share of each land-cover type that falls in the
high-SD tail. It records where the spread shows up, not why.
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
from plot_style import arial, use_arial

ROOT = Path(__file__).resolve().parent
PHENO = ROOT / "data/phenology/phenology_threshold_sl122_evi.nc"
CUBE = ROOT / "data/phenology/veg_cube.nc"
FINAL = ROOT / "figures/final"
FIG = ROOT / "figures"
HEIGHT = 1830
MIN_YEARS = 6
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


def within_type_rate(lum: np.ndarray, ok: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Fraction of each land-cover type whose pixels are in the high-SD set."""
    rates = []
    for code, _name in COVER:
        pool = ok & (lum == code)
        rates.append(float(np.mean(high[pool])) if np.any(pool) else np.nan)
    return np.array(rates, np.float64)


def map_panel(ax, image, extent, title, norm, cmap, font) -> object:
    show = np.ma.masked_invalid(image)
    im = ax.imshow(show, origin="upper", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
    med = float(np.nanmedian(image))
    ax.set_title(f"{title}\nmedian {med:.0f} d", fontproperties=font, fontsize=12, loc="left")
    ax.set_xlabel("Longitude", fontproperties=font)
    ax.set_ylabel("Latitude", fontproperties=font)
    ax.tick_params(labelsize=8)
    return im


def draw(sos_img, eos_img, extent, shares, sos_cut, eos_cut) -> None:
    use_arial()
    font = arial(11)
    small = arial(10)
    fig = plt.figure(figsize=(12.2, 8.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])
    ax_sos = fig.add_subplot(gs[0, 0])
    ax_eos = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[:, 1])
    finite = np.concatenate([sos_img[np.isfinite(sos_img)], eos_img[np.isfinite(eos_img)]])
    vmax = float(np.ceil(np.percentile(finite, 98) / 5.0) * 5.0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")
    norm = Normalize(0, vmax)
    im = map_panel(ax_sos, sos_img, extent, "(a)  Green-up (15%)", norm, cmap, font)
    map_panel(ax_eos, eos_img, extent, "Dormancy (50%)", norm, cmap, font)
    cbar = fig.colorbar(im, ax=[ax_sos, ax_eos], orientation="horizontal", fraction=0.04, pad=0.18)
    cbar.set_label("SD (days)", fontproperties=font, fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    labels = [name for _code, name in COVER]
    y = np.arange(len(labels))
    height = 0.36
    series = (
        (shares["sos"], "#1b7f4e", f"Green-up SD ≥ {sos_cut:.0f} d"),
        (shares["eos"], "#b86e00", f"Dormancy SD ≥ {eos_cut:.0f} d"),
    )
    for i, (values, color, label) in enumerate(series):
        ax_bar.barh(y + (0.5 - i) * height, 100 * values, height=height, color=color, label=label)
    ax_bar.axvline(10, color="#666666", lw=1.0, ls="--", label="10% of all pixels")
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontproperties=small, fontsize=11)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("High-SD pixels in type (%)", fontproperties=font)
    top = float(np.nanmax([100 * values for values, _color, _label in series]))
    ax_bar.set_xlim(0, max(28, np.ceil(top / 5.0) * 5.0 + 4))
    ax_bar.set_title("(b)  High-SD pixels within each type", fontproperties=font, fontsize=12, loc="left")
    ax_bar.legend(frameon=False, prop=arial(9), loc="upper right")
    ax_bar.tick_params(labelsize=9)
    path = FINAL / "10_phenology_interannual_sd.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"wrote {path}")


def main() -> None:
    row, col, sos, eos, lum_grid = load_dates()
    sos_sd = pixel_sd(sos)
    eos_sd = pixel_sd(eos)
    lum = lum_grid[row, col]
    meta = json.loads((ROOT / "data/meta.json").read_text())["bounds_lonlat"]
    extent = [meta["west"], meta["east"], meta["south"], meta["north"]]
    sos_ok = np.isfinite(sos_sd)
    eos_ok = np.isfinite(eos_sd)
    sos_cut = float(np.percentile(sos_sd[sos_ok], 90))
    eos_cut = float(np.percentile(eos_sd[eos_ok], 90))
    shares = {
        "sos": within_type_rate(lum, sos_ok, sos_ok & (sos_sd >= sos_cut)),
        "eos": within_type_rate(lum, eos_ok, eos_ok & (eos_sd >= eos_cut)),
    }
    for key, values in shares.items():
        text = "  ".join(f"{name} {100 * value:.1f}%" for (_code, name), value in zip(COVER, values))
        log(f"{key:4} {text}")
    draw(
        paint_pixels(row, col, sos_sd),
        paint_pixels(row, col, eos_sd),
        extent,
        shares,
        sos_cut,
        eos_cut,
    )
    out = FIG / "phenology_scale_variance.csv"
    pd.DataFrame({
        "type": [name for _code, name in COVER],
        "greenup_high_sd_in_type": shares["sos"],
        "dormancy_high_sd_in_type": shares["eos"],
    }).to_csv(out, index=False)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
