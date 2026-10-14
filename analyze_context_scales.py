#!/usr/bin/env python3
"""Spatial context of SOS and EOS across grid sizes.

Each vegetation pixel keeps its multi-year median date. Cells then average
those dates and record four surroundings, measured the same way at every scale:

- urban fraction of the land in the cell
- natural-vegetation fraction of that land
- mean distance from the land to the sea
- a heat-island residual: June–August station temperature, after removing
  elevation and distance to the sea, then interpolated to the cell

The residual is only as sharp as the station network, about 3 km.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from matplotlib import font_manager
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr

from analyze_driver_grids import CLIM, es_kpa, idw_weights, local_km, station_matrix

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
PHENO = ROOT / "data" / "phenology" / "phenology_threshold_sl122_evi.nc"
FONT = Path("/tmp/NotoSansSC-Regular.otf")
X0, Y0, PIX = 799980.0, 2500020.0, 30.0
GRIDS_M = (500, 1000, 2000, 5000, 10000, 20000)
URBAN = (1, 2, 3, 11, 21, 22, 23, 31, 41, 42, 43, 44)
GREEN = (71, 72, 73, 74)
MIN_VEG = 15
MIN_CELLS = 8
TO_XY = pyproj.Transformer.from_crs(4326, 32649, always_xy=True)
DRIVERS = (
    ("urban", "Built-up fraction"),
    ("green", "Green-cover fraction"),
    ("sea_km", "Distance to sea (km)"),
    ("uhi", "Heat-island residual (°C)"),
)


def fontprop():
    from plot_style import arial
    return arial(10)


def load_phenology() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(PHENO) as src:
        row = np.array(src.variables["row"][:], np.int32)
        col = np.array(src.variables["col"][:], np.int32)
        lum = np.array(src.variables["lum_code"][:], np.int16) if "lum_code" in src.variables else None
        sos = np.array(src.variables["sos15"][:], np.float32)
        eos = np.array(src.variables["eos50"][:], np.float32)
    if lum is None:
        with Dataset(ROOT / "data/phenology/veg_cube.nc") as src:
            lum = np.array(src.variables["lum_code"][:], np.int16)
    short = np.isfinite(sos) & np.isfinite(eos) & (eos <= sos + 30.0)
    sos[short] = np.nan
    eos[short] = np.nan
    enough = np.isfinite(sos).sum(axis=1) >= 5
    sos_med = np.nanmedian(sos, axis=1).astype(np.float32)
    eos_med = np.nanmedian(eos, axis=1).astype(np.float32)
    sos_med[~enough] = np.nan
    eos_med[~enough] = np.nan
    return lum, row, col, sos_med, eos_med


def summer_temperature() -> pd.DataFrame:
    days = pd.date_range("2015-01-01", "2025-12-31", freq="D")
    stations = pd.read_csv(CLIM / "stations.csv")
    codes = stations.loc[stations["temp"] == 1, "code"].tolist()
    values, kept = station_matrix(codes, "TEMP", days)
    jja = np.isin(days.month, (6, 7, 8))
    part = values[:, jja]
    good = np.isfinite(part).mean(axis=1) >= 0.7
    mean = np.nanmean(part, axis=1)
    meta = stations.set_index("code").loc[np.array(kept)[good]].copy()
    meta["t_jja"] = mean[good]
    meta["elev_m"] = pd.to_numeric(meta["elev_m"], errors="coerce")
    return meta.reset_index()


def sea_distance(lum: np.ndarray) -> np.ndarray:
    sea = lum <= 0
    return distance_transform_edt(~sea).astype(np.float32) * (PIX / 1000.0)


def sample_grid(dist: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    x, y = TO_XY.transform(lon, lat)
    col = np.rint((x - X0) / PIX - 0.5).astype(int)
    row = np.rint((Y0 - y) / PIX - 0.5).astype(int)
    out = np.full(lon.shape, np.nan)
    inside = (row >= 0) & (row < dist.shape[0]) & (col >= 0) & (col < dist.shape[1])
    out[inside] = dist[row[inside], col[inside]]
    return out


def heat_island(meta: pd.DataFrame, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sea = sample_grid(dist, meta["lon"].to_numpy(np.float64), meta["lat"].to_numpy(np.float64))
    elev = meta["elev_m"].to_numpy(np.float64)
    temp = meta["t_jja"].to_numpy(np.float64)
    ok = np.isfinite(sea) & np.isfinite(elev) & np.isfinite(temp)
    design = np.column_stack([np.ones(ok.sum()), elev[ok], sea[ok]])
    coef, *_ = np.linalg.lstsq(design, temp[ok], rcond=None)
    pred = coef[0] + coef[1] * elev + coef[2] * sea
    resid = np.full(temp.shape, np.nan)
    resid[ok] = temp[ok] - pred[ok]
    print(
        f"summer temp ~ elevation {coef[1]:+.4f} C/m, sea distance {coef[2]:+.4f} C/km, "
        f"stations {int(ok.sum())}",
        flush=True,
    )
    xy = local_km(meta["lon"].to_numpy(np.float64), meta["lat"].to_numpy(np.float64))
    return resid, temp, xy


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < MIN_CELLS or np.std(x[mask]) < 1e-8 or np.std(y[mask]) < 1e-8:
        return np.nan, np.nan, n
    r, p = pearsonr(x[mask], y[mask])
    return float(r), float(p), n


def standardize(columns: list[np.ndarray]) -> np.ndarray:
    out = []
    for values in columns:
        mu, sd = np.nanmean(values), np.nanstd(values)
        out.append((values - mu) / sd if sd > 0 else np.zeros_like(values))
    return np.column_stack(out)


def cells_at(lum, dist, row, col, sos, eos, grid_m, resid, temp, station_xy) -> pd.DataFrame:
    pix = max(1, int(round(grid_m / PIX)))
    land = lum > 0
    ys, xs = np.nonzero(land)
    key = (ys // pix) * 100000 + (xs // pix)
    uniq, inv = np.unique(key, return_inverse=True)
    n_land = np.bincount(inv)
    urban = np.isin(lum, URBAN)
    green = np.isin(lum, GREEN)
    n_urban = np.bincount(inv, weights=urban[ys, xs].astype(np.float64))
    n_green = np.bincount(inv, weights=green[ys, xs].astype(np.float64))
    sea = np.bincount(inv, weights=dist[ys, xs].astype(np.float64)) / np.maximum(n_land, 1)
    ok = np.isfinite(sos) & np.isfinite(eos)
    pkey = (row[ok] // pix) * 100000 + (col[ok] // pix)
    pos = np.searchsorted(uniq, pkey)
    matched = (pos < uniq.size) & (uniq[np.clip(pos, 0, uniq.size - 1)] == pkey)
    inv_p = np.full(pkey.shape, -1, np.int32)
    inv_p[matched] = pos[matched]
    use = inv_p >= 0
    n_veg = np.bincount(inv_p[use], minlength=uniq.size)
    sos_sum = np.bincount(inv_p[use], weights=sos[ok][use].astype(np.float64), minlength=uniq.size)
    eos_sum = np.bincount(inv_p[use], weights=eos[ok][use].astype(np.float64), minlength=uniq.size)
    keep = n_veg >= MIN_VEG
    cr, cc = uniq // 100000, uniq % 100000
    x = X0 + (cc * pix + (pix - 1) / 2.0) * PIX
    y = Y0 - (cr * pix + (pix - 1) / 2.0) * PIX
    lon, lat = pyproj.Transformer.from_crs(32649, 4326, always_xy=True).transform(x, y)
    cell_xy = local_km(np.asarray(lon, np.float64), np.asarray(lat, np.float64))
    finite_station = np.isfinite(resid) & np.isfinite(temp)
    neighbor, weight = idw_weights(cell_xy, station_xy[finite_station])
    uhi = interpolate_column(resid[finite_station], neighbor, weight)
    thermal = interpolate_column(temp[finite_station], neighbor, weight)
    frame = pd.DataFrame({
        "sos": sos_sum[keep] / n_veg[keep],
        "eos": eos_sum[keep] / n_veg[keep],
        "urban": n_urban[keep] / n_land[keep],
        "green": n_green[keep] / n_land[keep],
        "sea_km": sea[keep],
        "uhi": uhi[keep],
        "t_jja": thermal[keep],
        "n_veg": n_veg[keep],
        "iy": cr[keep],
        "ix": cc[keep],
    })
    return frame


def interpolate_column(values: np.ndarray, neighbor: np.ndarray, weight: np.ndarray) -> np.ndarray:
    picked = values[neighbor]
    valid = np.isfinite(picked)
    w = np.where(valid, weight, 0.0)
    denom = w.sum(axis=1)
    num = np.nansum(np.where(valid, picked, 0.0) * w, axis=1)
    return np.divide(num, denom, out=np.full(num.shape, np.nan), where=denom > 0)


def joint(frame: pd.DataFrame, target: str) -> dict[str, float]:
    predictors = ["urban", "sea_km", "uhi"]
    mask = np.isfinite(frame[target])
    for name in predictors:
        mask &= np.isfinite(frame[name])
    if int(mask.sum()) < MIN_CELLS:
        return {name: np.nan for name in predictors}
    y = frame.loc[mask, target].to_numpy(np.float64)
    y = (y - y.mean()) / y.std()
    x = standardize([frame.loc[mask, name].to_numpy(np.float64) for name in predictors])
    design = np.column_stack([np.ones(x.shape[0]), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {name: float(value) for name, value in zip(predictors, coef[1:])}


def draw_lines(table: pd.DataFrame, path: Path, font) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=140, sharey=True)
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.12)
    colors = {"urban": "#c0392b", "green": "#1e8449", "sea_km": "#1f618d", "uhi": "#d68910"}
    for ax, target, title in zip(axes, ("sos", "eos"), ("Green-up, 15%", "Dormancy, 50%")):
        for key, label in DRIVERS:
            block = table[(table["target"] == target) & (table["driver"] == key)].sort_values("grid_m")
            ax.plot(block["grid_m"] / 1000, block["r"], marker="o", color=colors[key], label=label, lw=1.6)
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5, 10, 20])
        ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
        ax.set_xlabel("Cell size (km)", fontproperties=font, fontsize=11)
        ax.set_title(title, fontproperties=font, fontsize=13, loc="left")
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("Correlation", fontproperties=font, fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, prop=font, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Correlation vs cell size. Green-up 15%, dormancy 50%.", fontproperties=font, fontsize=12)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def draw_scatter(frame: pd.DataFrame, path: Path, font) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.8), dpi=130)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.10, hspace=0.42, wspace=0.34)
    ylabels = ("Green-up (15%)", "Dormancy (50%)")
    for row, (target, ylabel) in enumerate((("sos", ylabels[0]), ("eos", ylabels[1]))):
        for col, (key, label) in enumerate(DRIVERS):
            ax = axes[row, col]
            x = frame[key].to_numpy(np.float64)
            y = frame[target].to_numpy(np.float64)
            r, _p, _n = corr(x, y)
            ax.scatter(x, y, s=12, color="#1f4e79", alpha=0.55, linewidths=0)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 2 and np.std(x[mask]) > 0:
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                xs = np.linspace(np.min(x[mask]), np.max(x[mask]), 40)
                ax.plot(xs, intercept + slope * xs, color="#c0392b", lw=1.4)
            letter = "abcdefgh"[row * 4 + col]
            ax.set_title(f"({letter})  {label}", fontproperties=font, fontsize=12, loc="left", pad=10)
            ax.annotate(
                f"r = {r:+.2f}",
                xy=(1, 1), xycoords="axes fraction",
                xytext=(0, 3), textcoords="offset points",
                ha="right", va="bottom",
                fontproperties=font, fontsize=11, color="#222222",
                annotation_clip=False,
            )
            if row == 1:
                ax.set_xlabel(label, fontproperties=font, fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontproperties=font, fontsize=12)
            ax.tick_params(labelsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_maps(frame: pd.DataFrame, path: Path, font) -> None:
    fields = (
        ("sos", "(a)  Green-up (15%)", "Day of year", "viridis"),
        ("eos", "(b)  Dormancy (50%)", "Day of year", "viridis"),
        ("urban", "(c)  Built-up fraction", "Fraction", "YlOrRd"),
        ("uhi", "(d)  Heat-island residual", "°C", "coolwarm"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.2), dpi=130)
    fig.subplots_adjust(top=0.96, wspace=0.22, hspace=0.18)
    for ax, (key, label, unit, cmap) in zip(axes.ravel(), fields):
        iy = frame["iy"].to_numpy()
        ix = frame["ix"].to_numpy()
        img = np.full((iy.max() + 1, ix.max() + 1), np.nan)
        img[iy, ix] = frame[key]
        vmax = np.nanpercentile(np.abs(img), 98) if key == "uhi" else None
        if key == "uhi":
            im = ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax, origin="upper")
        else:
            lo, hi = np.nanpercentile(img, [2, 98])
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, origin="upper")
        ax.set_title(label, fontproperties=font, fontsize=12, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(unit, fontproperties=font, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    font = fontprop()
    lum, row, col, sos, eos = load_phenology()
    dist = sea_distance(lum)
    meta = summer_temperature()
    resid, temp, station_xy = heat_island(meta, dist)
    order = np.argsort(resid)
    print("warmest residuals:")
    for i in order[-5:][::-1]:
        print(f"  {meta.loc[i, 'code']} {resid[i]:+.2f} C  elev {meta.loc[i, 'elev_m']}  t {temp[i]:.1f}")
    print("coolest residuals:")
    for i in order[:5]:
        if not np.isfinite(resid[i]):
            continue
        print(f"  {meta.loc[i, 'code']} {resid[i]:+.2f} C  elev {meta.loc[i, 'elev_m']}  t {temp[i]:.1f}")
    records = []
    stored = {}
    for grid in GRIDS_M:
        frame = cells_at(lum, dist, row, col, sos, eos, grid, resid, temp, station_xy)
        stored[grid] = frame
        print(f"\n{grid} m cells {len(frame)}", flush=True)
        for target in ("sos", "eos"):
            betas = joint(frame, target)
            for key, _label in DRIVERS:
                r, p, n = corr(frame[key].to_numpy(), frame[target].to_numpy())
                print(f"  {target} {key:7} r {r:+.3f} p {p:.3f} n {n}")
                records.append({
                    "grid_m": grid, "target": target, "driver": key,
                    "r": r, "p": p, "n": n, "beta": betas.get(key, np.nan),
                })
        print(
            "  driver cross r urban-green "
            f"{corr(frame['urban'], frame['green'])[0]:+.2f}  urban-uhi {corr(frame['urban'], frame['uhi'])[0]:+.2f}  "
            f"sea-uhi {corr(frame['sea_km'], frame['uhi'])[0]:+.2f}"
        )
    table = pd.DataFrame(records)
    FIG.mkdir(parents=True, exist_ok=True)
    table.to_csv(FIG / "context_scale_correlations.csv", index=False)
    draw_lines(table, FIG / "context_scale_correlations.png", font)
    final = FIG / "final"
    final.mkdir(parents=True, exist_ok=True)
    draw_scatter(stored[2000], FIG / "context_scatter_2km.png", font)
    draw_scatter(stored[2000], final / "context_scatter_2km.png", font)
    draw_maps(stored[1000], FIG / "context_maps_1km.png", font)
    draw_maps(stored[1000], final / "context_maps_1km.png", font)
    print("wrote figures/context_scale_correlations.png")


if __name__ == "__main__":
    main()
