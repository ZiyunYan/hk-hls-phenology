#!/usr/bin/env python3
"""Year-to-year spread of green-up and dormancy at 30 m, 1 km, and 2 km.

The maps use the standard deviation across years. A 1 km or 2 km value is the
spread of that cell's yearly mean date. Pixels need six valid years. A cell-year
mean needs 40 dated vegetation pixels, and the cell needs six such years.
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
from scipy.stats import pearsonr

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
MIN_CELL = 40
GREEN = (71, 72, 73, 74)
URBAN = (1, 2, 3, 11, 21, 22, 23, 31, 41, 42, 43, 44)
COVER = {71: "Woodland", 72: "Shrubland", 73: "Grassland", 74: "Mangrove"}


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


def cell_keys(row: np.ndarray, col: np.ndarray, pix: int) -> np.ndarray:
    return (row.astype(np.int64) // pix) * 100000 + (col.astype(np.int64) // pix)


def grid_sd(row: np.ndarray, col: np.ndarray, dates: np.ndarray, pix: int) -> tuple[np.ndarray, np.ndarray]:
    """Interannual SD of the cell-year mean, and within-cell SD of pixel medians."""
    key = cell_keys(row, col, pix)
    uniq, inv = np.unique(key, return_inverse=True)
    n_year = dates.shape[1]
    mean = np.full((uniq.size, n_year), np.nan, np.float64)
    for year in range(n_year):
        values = dates[:, year]
        ok = np.isfinite(values)
        if not np.any(ok):
            continue
        count = np.bincount(inv[ok], minlength=uniq.size)
        total = np.bincount(inv[ok], weights=values[ok].astype(np.float64), minlength=uniq.size)
        good = count >= MIN_CELL
        mean[good, year] = total[good] / count[good]
    enough = np.isfinite(mean).sum(axis=1) >= MIN_YEARS
    temporal = np.full(uniq.size, np.nan, np.float32)
    temporal[enough] = np.nanstd(mean[enough], axis=1)

    count = np.isfinite(dates).sum(axis=1)
    kept = count >= MIN_YEARS
    med = np.nanmedian(dates, axis=1)
    use = kept & np.isfinite(med)
    n = np.bincount(inv[use], minlength=uniq.size).astype(np.float64)
    total = np.bincount(inv[use], weights=med[use].astype(np.float64), minlength=uniq.size)
    total2 = np.bincount(inv[use], weights=(med[use].astype(np.float64) ** 2), minlength=uniq.size)
    spatial = np.full(uniq.size, np.nan, np.float32)
    many = n >= MIN_CELL
    var = (total2[many] - total[many] ** 2 / n[many]) / (n[many] - 1.0)
    spatial[many] = np.sqrt(np.clip(var, 0, None))
    return uniq, temporal, spatial, inv, med, kept


def paint_cells(uniq: np.ndarray, values: np.ndarray, pix: int, veg: np.ndarray) -> np.ndarray:
    iy, ix = np.indices((HEIGHT, HEIGHT))
    key = (iy // pix).astype(np.int64) * 100000 + (ix // pix)
    pos = np.searchsorted(uniq, key.ravel())
    pos = np.clip(pos, 0, uniq.size - 1)
    matched = uniq[pos] == key.ravel()
    out = np.full(key.size, np.nan, np.float32)
    take = matched & np.isfinite(values[pos])
    out[take] = values[pos[take]]
    image = out.reshape(HEIGHT, HEIGHT)
    image[~veg] = np.nan
    return image


def paint_pixels(row: np.ndarray, col: np.ndarray, values: np.ndarray) -> np.ndarray:
    image = np.full((HEIGHT, HEIGHT), np.nan, np.float32)
    ok = np.isfinite(values)
    image[row[ok], col[ok]] = values[ok]
    return image


def variance_share(row, col, dates, pix: int) -> float:
    """Fraction of pixel-date variance that sits inside cells, pooled over years."""
    key = cell_keys(row, col, pix)
    within = 0.0
    total = 0.0
    for year in range(dates.shape[1]):
        values = dates[:, year]
        ok = np.isfinite(values)
        if int(ok.sum()) < 100:
            continue
        v = values[ok].astype(np.float64)
        k = key[ok]
        _, inv = np.unique(k, return_inverse=True)
        n = np.bincount(inv).astype(np.float64)
        sm = np.bincount(inv, weights=v)
        grand = v.mean()
        cell = sm / n
        total += float(np.sum((v - grand) ** 2))
        within += float(np.sum((v - cell[inv]) ** 2))
    return within / total if total > 0 else np.nan


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


def group_line(name: str, mask: np.ndarray, fields: dict[str, np.ndarray]) -> None:
    n = int(mask.sum())
    parts = [f"{name:16} n {n:7d}"]
    for key in ("elev", "slope", "northness", "sea_m", "urban_m"):
        parts.append(f"{key} {np.nanmedian(fields[key][mask]):7.1f}")
    lum = fields["lum"][mask]
    for code, label in COVER.items():
        parts.append(f"{label} {np.mean(lum == code):5.1%}")
    log("  " + "  ".join(parts))


def summarize(label: str, values: np.ndarray) -> None:
    v = values[np.isfinite(values)]
    qs = np.percentile(v, [10, 50, 90])
    log(f"{label:22} n {v.size:7d}  p10 {qs[0]:5.1f}  median {qs[1]:5.1f}  p90 {qs[2]:5.1f}")


def draw(images: list[tuple[str, np.ndarray]], extent: list[float]) -> None:
    use_arial()
    font = arial(11)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.8), layout="constrained")
    finite = np.concatenate([img[np.isfinite(img)] for _, img in images])
    vmax = float(np.ceil(np.percentile(finite, 98) / 5.0) * 5.0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")
    last = None
    for ax, (title, image) in zip(axes.ravel(), images):
        show = np.ma.masked_invalid(image)
        last = ax.imshow(
            show, origin="upper", extent=extent, cmap=cmap,
            norm=Normalize(0, vmax), interpolation="nearest",
        )
        med = float(np.nanmedian(image))
        ax.set_title(f"{title}\nmedian {med:.0f} d", fontproperties=font, fontsize=12, loc="left")
        ax.set_xlabel("Longitude", fontproperties=font)
        ax.set_ylabel("Latitude", fontproperties=font)
        ax.tick_params(labelsize=8)
    cbar = fig.colorbar(last, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Interannual SD (days)", fontproperties=font, fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    FINAL.mkdir(parents=True, exist_ok=True)
    path = FINAL / "10_phenology_interannual_sd.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"wrote {path}")


def main() -> None:
    row, col, sos, eos, lum = load_dates()
    veg = np.isin(lum, GREEN)
    meta = json.loads((ROOT / "data/meta.json").read_text())["bounds_lonlat"]
    extent = [meta["west"], meta["east"], meta["south"], meta["north"]]
    panels = []
    records = []
    fields = None
    for target, dates in (("Green-up (15%)", sos), ("Dormancy (50%)", eos)):
        sd = pixel_sd(dates)
        summarize(f"{target} 30 m", sd)
        panels.append((f"({ 'ad'[target.startswith('D')] })  {target}, 30 m", paint_pixels(row, col, sd)))
        # fix letters after both rows are known
        for pix, scale, letter_sos, letter_eos in ((33, "1 km", "b", "e"), (67, "2 km", "c", "f")):
            uniq, temporal, spatial, inv, med, kept = grid_sd(row, col, dates, pix)
            summarize(f"{target} {scale} temporal", temporal)
            summarize(f"{target} {scale} within cell", spatial)
            share = variance_share(row, col, dates, pix)
            log(f"{target} {scale} variance inside cells {share:.2%}")
            letter = letter_eos if target.startswith("D") else letter_sos
            panels.append((f"({letter})  {target}, {scale}", paint_cells(uniq, temporal, pix, veg)))
            wide = spatial[np.isfinite(spatial)]
            records.append({
                "target": target, "scale": scale,
                "within_median_d": float(np.median(wide)),
                "within_p90_d": float(np.percentile(wide, 90)),
                "frac_within_gt10": float(np.mean(wide > 10)),
                "frac_within_gt15": float(np.mean(wide > 15)),
                "variance_inside": share,
            })
            if target.startswith("G") and pix == 33:
                cell_med = np.full(row.shape, np.nan, np.float32)
                n = np.bincount(inv, minlength=uniq.size)
                # median of dated pixels is not the bincount mean; use the pixel median's cell mean
                use = kept & np.isfinite(med)
                total = np.bincount(inv[use], weights=med[use].astype(np.float64), minlength=uniq.size)
                nn = np.bincount(inv[use], minlength=uniq.size).astype(np.float64)
                good = nn >= MIN_CELL
                centre = np.full(uniq.size, np.nan, np.float32)
                centre[good] = total[good] / nn[good]
                cell_med[use] = centre[inv[use]]
                fields = topography(row, col, lum)
                fields["sos_sd"] = sd
                fields["dev"] = np.abs(med - cell_med)
                fields["kept"] = kept & np.isfinite(sd)
        if target.startswith("D"):
            fields["eos_sd"] = sd
    # Letters in storage order are wrong because each target appends 30 m then scales.
    # Rebuild in row-major a–f.
    ordered = [panels[0], panels[1], panels[2], panels[3], panels[4], panels[5]]
    letters = "abcdef"
    ordered = [(f"({letters[i]})  {title.split('  ', 1)[1]}", image) for i, (title, image) in enumerate(ordered)]
    draw(ordered, extent)

    kept = fields["kept"]
    sd = fields["sos_sd"][kept]
    edge = np.nanpercentile(sd, 90)
    mid = np.nanpercentile(sd, 50)
    unstable = kept.copy()
    unstable[kept] = sd >= edge
    stable = kept.copy()
    stable[kept] = sd <= mid
    log(f"unstable SOS SD >= {edge:.1f} d (p90), stable <= {mid:.1f} d (p50)")
    both = kept & np.isfinite(fields["eos_sd"])
    r, p = pearsonr(fields["sos_sd"][both], fields["eos_sd"][both])
    log(f"pixel SOS SD vs EOS SD  r {r:+.2f}  p {p:.3g}  n {int(both.sum())}")
    group_line("unstable", unstable, fields)
    group_line("stable", stable, fields)
    log(f"unstable |date - 1 km mean| median {np.nanmedian(fields['dev'][unstable]):.1f} d")
    log(f"stable   |date - 1 km mean| median {np.nanmedian(fields['dev'][stable]):.1f} d")
    table = pd.DataFrame(records)
    out = FIG / "phenology_scale_variance.csv"
    table.to_csv(out, index=False)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
