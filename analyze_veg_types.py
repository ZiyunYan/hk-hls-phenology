#!/usr/bin/env python3
"""SOS and EOS by Planning Department vegetation class.

Classes are the 2024 land-use code on pixels that were natural vegetation in
both 2018 and 2024: woodland 71, shrubland 72, grassland 73, mangrove 74.
Climate is the same station-mean series used for the territory correlations.
Surroundings are measured on 2 km cells, and dates are also compared inside
the same cell so location is shared.
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
from scipy.stats import linregress, pearsonr, theilslopes

from analyze_context_scales import (
    heat_island,
    idw_weights,
    local_km,
    sea_distance,
    summer_temperature,
)
from analyze_driver_grids import X0, Y0, PIX
from plot_driver_scatters import territory_climate

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
PHENO = ROOT / "data" / "phenology" / "phenology_threshold_sl122_evi.nc"
CUBE = ROOT / "data" / "phenology" / "veg_cube.nc"
FONT = Path("/tmp/NotoSansSC-Regular.otf")
GRID_M = 2000
MIN_PIX = 15
MIN_YEARS = 7
CLASSES = (
    (71, "wood", "林地", "#1b7f4e"),
    (72, "shrub", "灌木", "#b08900"),
    (73, "grass", "草地", "#d35400"),
    (74, "mangrove", "红树林", "#1f618d"),
)
PAIRS = (("wood", "shrub", "林地 − 灌木"), ("wood", "grass", "林地 − 草地"), ("shrub", "grass", "灌木 − 草地"))
CLIMATE = (
    ("vpd_mam", "sos", "3–5 月水汽压差与返青", "水汽压差 (kPa)"),
    ("temp_son", "eos", "9–11 月气温与休眠", "气温 (°C)"),
    ("temp_jja", "eos", "6–8 月气温与休眠", "气温 (°C)"),
)
CONTEXT = (("urban", "建成区"), ("green", "绿地"), ("sea_km", "离海"), ("uhi", "热岛"))


def fontprop():
    if FONT.is_file():
        font_manager.fontManager.addfont(str(FONT))
        return font_manager.FontProperties(fname=str(FONT))
    return font_manager.FontProperties(family="DejaVu Sans")


def load() -> dict[str, np.ndarray]:
    with Dataset(PHENO) as src:
        years = np.array(src.variables["year"][:], np.int32)
        row = np.array(src.variables["row"][:], np.int32)
        col = np.array(src.variables["col"][:], np.int32)
        sos = np.array(src.variables["sos15"][:], np.float32)
        eos = np.array(src.variables["eos50"][:], np.float32)
    with Dataset(CUBE) as src:
        lum = np.array(src.variables["lum_code"][:], np.int16)
    short = np.isfinite(sos) & np.isfinite(eos) & (eos <= sos + 30.0)
    sos = sos.copy()
    eos = eos.copy()
    sos[short] = np.nan
    eos[short] = np.nan
    return {"years": years, "row": row, "col": col, "sos": sos, "eos": eos, "lum": lum, "code": lum[row, col]}


def yearly_median(values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmedian(values, axis=0)


def pixel_median(values: np.ndarray) -> np.ndarray:
    enough = np.isfinite(values).sum(axis=1) >= 5
    with np.errstate(all="ignore"):
        med = np.nanmedian(values, axis=1)
    med[~enough] = np.nan
    return med


def trend(years: np.ndarray, values: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(values)
    if int(mask.sum()) < 6:
        return {"slope": np.nan, "p": np.nan, "theil": np.nan, "theil_lo": np.nan, "theil_hi": np.nan}
    slope, _, _, p, _ = linregress(years[mask].astype(float), values[mask])
    theil, _, lo, hi = theilslopes(values[mask], years[mask].astype(float))
    return {"slope": float(slope), "p": float(p), "theil": float(theil), "theil_lo": float(lo), "theil_hi": float(hi)}


def pearson(x: np.ndarray, y: np.ndarray, min_n: int = 6) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_n or np.std(x[mask]) < 1e-8 or np.std(y[mask]) < 1e-8:
        return np.nan, np.nan, n
    r, p = pearsonr(x[mask], y[mask])
    return float(r), float(p), n


def cell_index(row: np.ndarray, col: np.ndarray, pix: int) -> tuple[np.ndarray, np.ndarray]:
    key = (row // pix).astype(np.int64) * 100000 + (col // pix).astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    return uniq, inv


def bincount_mean(inv: np.ndarray, values: np.ndarray, n_cells: int, min_pix: int) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(values)
    count = np.bincount(inv[ok], minlength=n_cells)
    total = np.bincount(inv[ok], weights=values[ok].astype(np.float64), minlength=n_cells)
    mean = np.full(n_cells, np.nan)
    keep = count >= min_pix
    mean[keep] = total[keep] / count[keep]
    return mean, count


def land_frame(lum: np.ndarray, dist: np.ndarray, resid: np.ndarray, station_xy: np.ndarray, pix: int) -> pd.DataFrame:
    land = lum > 0
    ys, xs = np.nonzero(land)
    key = (ys // pix).astype(np.int64) * 100000 + (xs // pix).astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    n_land = np.bincount(inv)
    urban = np.isin(lum, (1, 2, 3, 11, 21, 22, 23, 31, 41, 42, 43, 44))
    green = np.isin(lum, (71, 72, 73, 74))
    n_urban = np.bincount(inv, weights=urban[ys, xs].astype(np.float64))
    n_green = np.bincount(inv, weights=green[ys, xs].astype(np.float64))
    sea = np.bincount(inv, weights=dist[ys, xs].astype(np.float64)) / np.maximum(n_land, 1)
    cr, cc = uniq // 100000, uniq % 100000
    x = X0 + (cc * pix + (pix - 1) / 2.0) * PIX
    y = Y0 - (cr * pix + (pix - 1) / 2.0) * PIX
    lon, lat = pyproj.Transformer.from_crs(32649, 4326, always_xy=True).transform(x, y)
    cell_xy = local_km(np.asarray(lon, np.float64), np.asarray(lat, np.float64))
    finite = np.isfinite(resid)
    neighbor, weight = idw_weights(cell_xy, station_xy[finite])
    picked = resid[finite][neighbor]
    valid = np.isfinite(picked)
    w = np.where(valid, weight, 0.0)
    denom = w.sum(axis=1)
    num = np.nansum(np.where(valid, picked, 0.0) * w, axis=1)
    uhi = np.divide(num, denom, out=np.full(num.shape, np.nan), where=denom > 0)
    return pd.DataFrame({
        "key": uniq,
        "urban": n_urban / n_land,
        "green": n_green / n_land,
        "sea_km": sea,
        "uhi": uhi,
    })


def type_cell_dates(row, col, sos_med, eos_med, pix: int) -> pd.DataFrame:
    finite = np.isfinite(sos_med) & np.isfinite(eos_med)
    uniq, inv = cell_index(row[finite], col[finite], pix)
    sos_mean, count = bincount_mean(inv, sos_med[finite], uniq.size, MIN_PIX)
    eos_mean, _ = bincount_mean(inv, eos_med[finite], uniq.size, MIN_PIX)
    keep = count >= MIN_PIX
    return pd.DataFrame({"key": uniq[keep], "sos": sos_mean[keep], "eos": eos_mean[keep], "n": count[keep]})


def cell_climate(row, col, values, climate, pix: int) -> dict[str, float]:
    """How cells of one class track one territory climate series."""
    uniq, inv = cell_index(row, col, pix)
    columns = []
    for year in range(values.shape[1]):
        mean, _ = bincount_mean(inv, values[:, year], uniq.size, MIN_PIX)
        columns.append(mean)
    grid = np.column_stack(columns)
    per_cell = []
    for i in range(grid.shape[0]):
        r, _, n = pearson(climate, grid[i], min_n=MIN_YEARS)
        if n >= MIN_YEARS and np.isfinite(r):
            per_cell.append(r)
    per_cell = np.asarray(per_cell, np.float64)
    anomaly = grid - np.nanmean(grid, axis=1, keepdims=True)
    flat_y = anomaly.ravel()
    flat_x = np.broadcast_to(climate, grid.shape).ravel()
    r_anom, p_anom, n_anom = pearson(flat_x, flat_y, min_n=30)
    return {
        "n_cells": int(per_cell.size),
        "median_r": float(np.median(per_cell)) if per_cell.size else np.nan,
        "frac_positive": float(np.mean(per_cell > 0)) if per_cell.size else np.nan,
        "anomaly_r": r_anom,
        "anomaly_p": p_anom,
        "anomaly_n": n_anom,
    }


def bxp_stats(values: np.ndarray) -> dict:
    clean = values[np.isfinite(values)]
    p10, q1, med, q3, p90 = np.percentile(clean, [10, 25, 50, 75, 90])
    return {"med": med, "q1": q1, "q3": q3, "whislo": p10, "whishi": p90, "fliers": []}


def draw_dates(yearly: pd.DataFrame, pixel: dict[str, dict[str, np.ndarray]], path: Path, font) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), dpi=140)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.08, hspace=0.38, wspace=0.22)
    for ax, field, title in (
        (axes[0, 0], "sos", "每年中位返青"),
        (axes[0, 1], "eos", "每年中位休眠"),
    ):
        for code, key, label, color in CLASSES:
            block = yearly[yearly["veg"] == key]
            ax.plot(block["year"], block[field], marker="o", ms=4.5, lw=1.5, color=color, label=label)
        ax.set_title(title, fontproperties=font, fontsize=12, loc="left")
        ax.set_xlabel("年", fontproperties=font)
        ax.set_ylabel("日序", fontproperties=font)
        ax.set_xticks(sorted(yearly["year"].unique()))
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, prop=font, bbox_to_anchor=(0.55, 0.98))
    for ax, field, title in (
        (axes[1, 0], "sos", "像元多年中位返青"),
        (axes[1, 1], "eos", "像元多年中位休眠"),
    ):
        stats = [bxp_stats(pixel[key][field]) for _, key, _, _ in CLASSES]
        ax.bxp(stats, showfliers=False, widths=0.55, patch_artist=True,
               boxprops={"facecolor": "#f4f1ea", "edgecolor": "#333333"},
               medianprops={"color": "#333333"})
        for i, (_, _, _, color) in enumerate(CLASSES, start=1):
            ax.plot([i - 0.22, i + 0.22], [stats[i - 1]["med"], stats[i - 1]["med"]], color=color, lw=2.4, solid_capstyle="butt")
        ax.set_xticks(range(1, 5), [label for _, _, label, _ in CLASSES], fontproperties=font)
        ax.set_title(title + "（须线为 10%–90%）", fontproperties=font, fontsize=12, loc="left")
        ax.set_ylabel("日序", fontproperties=font)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def draw_climate(frame: pd.DataFrame, path: Path, font) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(12.4, 11.2), dpi=130, sharex="col")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.06, hspace=0.38, wspace=0.28)
    for row, (_, key, label, color) in enumerate(CLASSES):
        block = frame[frame["veg"] == key]
        years = block["year"].to_numpy()
        for col, (xname, yname, title, xlabel) in enumerate(CLIMATE):
            ax = axes[row, col]
            x = block[xname].to_numpy(np.float64)
            y = block[yname].to_numpy(np.float64)
            r, p, n = pearson(x, y)
            late = years >= 2016
            r2, p2, _ = pearson(x[late], y[late])
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 2 and np.std(x[mask]) > 0:
                slope, intercept, _, _, _ = linregress(x[mask], y[mask])
                xs = np.linspace(np.min(x[mask]), np.max(x[mask]), 40)
                ax.plot(xs, intercept + slope * xs, color=color, lw=1.5)
            highlight = (years == 2018) & (yname == "eos")
            ax.scatter(x[~highlight], y[~highlight], s=22, color=color, zorder=3)
            if np.any(highlight):
                ax.scatter(x[highlight], y[highlight], s=28, color="#d62728", zorder=4)
            ax.set_title(f"{label}  {title}\nr = {r:+.2f}（p = {p:.3f}）  去掉 2015  r = {r2:+.2f}",
                         fontproperties=font, fontsize=9, loc="left")
            if row == 3:
                ax.set_xlabel(xlabel, fontproperties=font, fontsize=9)
            ax.set_ylabel("返青（日序）" if col == 0 else "休眠（日序）", fontproperties=font, fontsize=9)
            ax.tick_params(labelsize=8)
            del n
    fig.suptitle("每一类的当年中位日期对站点平均气候。红点是 2018 年的休眠。", fontproperties=font, fontsize=13, x=0.07, ha="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def draw_place(paired: pd.DataFrame, path: Path, font) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=140)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.16, wspace=0.28)
    for ax, field, title in ((axes[0], "sos", "同一格子内的返青差"), (axes[1], "eos", "同一格子内的休眠差")):
        stats, labels = [], []
        for a, b, label in PAIRS:
            hit = paired[(paired["a"] == a) & (paired["b"] == b)]
            values = hit[field].to_numpy(np.float64)
            stats.append(bxp_stats(values))
            labels.append(f"{label}\nn = {len(hit)}")
        ax.bxp(stats, showfliers=False, widths=0.55, patch_artist=True,
               boxprops={"facecolor": "#f4f1ea", "edgecolor": "#333333"},
               medianprops={"color": "#333333", "lw": 1.6})
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.set_xticks(range(1, 4), labels, fontproperties=font, fontsize=9)
        ax.set_title(title, fontproperties=font, fontsize=12, loc="left")
        ax.set_ylabel("日差（须线 10%–90%）", fontproperties=font, fontsize=10)
    fig.suptitle("同一个 2 公里格子里，两类植被多年平均日期之差。正值表示前一类更晚。", fontproperties=font, fontsize=12, x=0.08, ha="left")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def draw_context(context: pd.DataFrame, path: Path, font) -> None:
    colors = {"urban": "#c0392b", "green": "#1e8449", "sea_km": "#1f618d", "uhi": "#d68910"}
    types = [key for _, key, _, _ in CLASSES]
    counts = {}
    for key in types:
        hit = context[(context["veg"] == key) & (context["target"] == "sos")]
        counts[key] = int(hit["n"].iloc[0]) if len(hit) else 0
    labels = [f"{label}\nn = {counts[key]}" for _, key, label, _ in CLASSES]
    x = np.arange(len(types))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), dpi=140, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.22, wspace=0.12)
    for ax, target, title in ((axes[0], "sos", "返青"), (axes[1], "eos", "休眠")):
        for i, (driver, label) in enumerate(CONTEXT):
            vals = []
            for key in types:
                hit = context[(context["veg"] == key) & (context["target"] == target) & (context["driver"] == driver)]
                vals.append(float(hit["r"].iloc[0]) if len(hit) else np.nan)
            ax.bar(x + (i - 1.5) * width, vals, width=width, color=colors[driver], label=label)
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.set_xticks(x, labels, fontproperties=font, fontsize=9)
        ax.set_ylim(-0.75, 0.75)
        ax.set_title(title, fontproperties=font, fontsize=12, loc="left")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("与格子平均日期的相关 r", fontproperties=font, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, prop=font, bbox_to_anchor=(0.53, 0.01))
    fig.suptitle("2 公里格子。周边是格子里的土地利用和热岛残差，日期只取这一类植被。", fontproperties=font, fontsize=12, x=0.08, ha="left")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    font = fontprop()
    data = load()
    years = data["years"]
    code = data["code"]
    print("class counts", {int(k): int(v) for k, v in zip(*np.unique(code, return_counts=True))}, flush=True)
    climate = territory_climate(years)
    all_sos = yearly_median(data["sos"])
    all_eos = yearly_median(data["eos"])
    for xname, yname, title, _ in CLIMATE:
        series = all_sos if yname == "sos" else all_eos
        r, p, n = pearson(climate[xname], series)
        print(f"all pixels {title}: r {r:+.3f} p {p:.3f} n {n}", flush=True)

    yearly_rows = []
    summary_rows = []
    climate_rows = []
    cell_rows = []
    pixel = {}
    cell_dates = {}
    pix = max(1, int(round(GRID_M / PIX)))
    for class_code, key, label, _color in CLASSES:
        mask = code == class_code
        sos = data["sos"][mask]
        eos = data["eos"][mask]
        sos_med = pixel_median(sos)
        eos_med = pixel_median(eos)
        pixel[key] = {"sos": sos_med, "eos": eos_med}
        finite = np.isfinite(sos_med) & np.isfinite(eos_med)
        summary_rows.append({
            "veg": key,
            "label": label,
            "code": class_code,
            "n_pixels": int(mask.sum()),
            "n_median": int(finite.sum()),
            "sos_p10": float(np.nanpercentile(sos_med, 10)),
            "sos_p50": float(np.nanpercentile(sos_med, 50)),
            "sos_p90": float(np.nanpercentile(sos_med, 90)),
            "eos_p10": float(np.nanpercentile(eos_med, 10)),
            "eos_p50": float(np.nanpercentile(eos_med, 50)),
            "eos_p90": float(np.nanpercentile(eos_med, 90)),
            "gsl_p50": float(np.nanpercentile(eos_med[finite] - sos_med[finite], 50)),
        })
        sos_year = yearly_median(sos)
        eos_year = yearly_median(eos)
        for i, year in enumerate(years):
            yearly_rows.append({
                "veg": key,
                "year": int(year),
                "sos": float(sos_year[i]),
                "eos": float(eos_year[i]),
                "gsl": float(eos_year[i] - sos_year[i]),
                "sos_valid": float(np.isfinite(sos[:, i]).mean()),
                "eos_valid": float(np.isfinite(eos[:, i]).mean()),
                **{name: float(values[i]) for name, values in climate.items()},
            })
        for field, series in (("sos", sos_year), ("eos", eos_year)):
            stat = trend(years, series)
            stat_late = trend(years[years >= 2016], series[years >= 2016])
            print(f"{label} {field} slope {stat['slope']:+.2f} d/y p {stat['p']:.2f} "
                  f"theil {stat['theil']:+.2f} [{stat['theil_lo']:+.2f}, {stat['theil_hi']:+.2f}] "
                  f"drop2015 slope {stat_late['slope']:+.2f} p {stat_late['p']:.2f}", flush=True)
        for xname, yname, title, _ in CLIMATE:
            series = sos_year if yname == "sos" else eos_year
            driver = climate[xname]
            for version, keep in (("2015-2025", np.ones(len(years), bool)), ("2016-2025", years >= 2016), ("drop2018", years != 2018)):
                r, p, n = pearson(driver[keep], series[keep])
                climate_rows.append({"veg": key, "label": label, "pair": f"{xname}_{yname}", "version": version, "r": r, "p": p, "n": n})
            cells = cell_climate(data["row"][mask], data["col"][mask], data["sos"][mask] if yname == "sos" else data["eos"][mask], driver, pix)
            cell_rows.append({"veg": key, "pair": f"{xname}_{yname}", **cells})
            print(f"{label} {title}: cells {cells['n_cells']} median r {cells['median_r']:+.2f} "
                  f"positive {cells['frac_positive']:.2f} anomaly {cells['anomaly_r']:+.2f}", flush=True)
        cell_dates[key] = type_cell_dates(data["row"][mask], data["col"][mask], sos_med, eos_med, pix)
        print(f"{label} 2 km cells {len(cell_dates[key])}", flush=True)

    dist = sea_distance(data["lum"])
    meta = summer_temperature()
    resid, _temp, station_xy = heat_island(meta, dist)
    land = land_frame(data["lum"], dist, resid, station_xy, pix)
    context_rows = []
    for class_code, key, label, _color in CLASSES:
        merged = cell_dates[key].merge(land, on="key", how="inner")
        print(f"{label} context cells {len(merged)} urban-green {pearson(merged['urban'], merged['green'])[0]:+.2f}", flush=True)
        for target in ("sos", "eos"):
            for driver, _name in CONTEXT:
                r, p, n = pearson(merged[driver].to_numpy(), merged[target].to_numpy(), min_n=8)
                context_rows.append({"veg": key, "label": label, "target": target, "driver": driver, "r": r, "p": p, "n": n})
                print(f"  {target} {driver:7} r {r:+.3f} p {p:.3f} n {n}")

    paired_rows = []
    for a, b, label in PAIRS:
        merged = cell_dates[a].merge(cell_dates[b], on="key", suffixes=("_a", "_b"))
        if merged.empty:
            continue
        sos_diff = merged["sos_a"] - merged["sos_b"]
        eos_diff = merged["eos_a"] - merged["eos_b"]
        print(f"{label} cells {len(merged)} SOS median {np.median(sos_diff):+.1f} "
              f"later {np.mean(sos_diff > 0):.2f}  EOS median {np.median(eos_diff):+.1f} later {np.mean(eos_diff > 0):.2f}", flush=True)
        for i in range(len(merged)):
            paired_rows.append({
                "a": a, "b": b, "label": label, "key": int(merged["key"].iloc[i]),
                "sos": float(sos_diff.iloc[i]), "eos": float(eos_diff.iloc[i]),
            })

    yearly = pd.DataFrame(yearly_rows)
    summary = pd.DataFrame(summary_rows)
    climate_table = pd.DataFrame(climate_rows)
    cell_table = pd.DataFrame(cell_rows)
    context = pd.DataFrame(context_rows)
    paired = pd.DataFrame(paired_rows)
    FIG.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(FIG / "vegtype_yearly.csv", index=False)
    summary.to_csv(FIG / "vegtype_summary.csv", index=False)
    climate_table.to_csv(FIG / "vegtype_climate.csv", index=False)
    cell_table.to_csv(FIG / "vegtype_cell_climate.csv", index=False)
    context.to_csv(FIG / "vegtype_context.csv", index=False)
    paired.to_csv(FIG / "vegtype_paired_cells.csv", index=False)
    draw_dates(yearly, pixel, FIG / "vegtype_dates.png", font)
    draw_climate(yearly, FIG / "vegtype_climate.png", font)
    draw_place(paired, FIG / "vegtype_place.png", font)
    draw_context(context, FIG / "vegtype_context.png", font)
    print(summary.to_string(index=False), flush=True)
    print("wrote vegtype_dates.png vegtype_climate.png vegtype_place.png vegtype_context.png", flush=True)


if __name__ == "__main__":
    main()
