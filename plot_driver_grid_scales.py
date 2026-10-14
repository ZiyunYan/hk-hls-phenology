#!/usr/bin/env python3
"""Phenology–climate scatters from the whole territory down to 1 km cells.

Each finer panel uses cell-years: one point per grid cell and year. Climate is
inverse-distance weighted from HKO stations onto the cell center. The fit is
the pooled cell-year correlation. The annotation also gives the within-cell
anomaly correlation, which repeats the year-to-year relationship across cells
without the spatial climatology.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from netCDF4 import Dataset
from scipy.stats import pearsonr

from analyze_driver_grids import (
    CLIM,
    MIN_PIXELS,
    MIN_YEARS,
    PIX,
    X0,
    Y0,
    cell_r,
    es_kpa,
    idw_weights,
    interpolate,
    local_km,
    station_matrix,
)
from analyze_driver_grids import TO_LL

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
PHENO = ROOT / "data" / "phenology" / "phenology_threshold_sl122_evi.nc"
GRIDS = (10000, 5000, 2000, 1000)
PAIRS = (
    ("vpd_mam", "sos", "Mar–May VPD and green-up", "VPD (kPa)", "Green-up (day)"),
    ("temp_son", "eos", "Sep–Nov temperature and dormancy", "Temperature (°C)", "Dormancy (day)"),
    ("temp_jja", "eos", "Jun–Aug temperature and dormancy", "Temperature (°C)", "Dormancy (day)"),
)


def season_station(values: np.ndarray, days: pd.DatetimeIndex, years: np.ndarray, months: tuple[int, ...]) -> np.ndarray:
    """values (stations, days) -> (stations, years) seasonal means."""
    out = np.full((values.shape[0], len(years)), np.nan)
    for j, year in enumerate(years):
        take = (days.year == int(year)) & np.isin(days.month, months)
        part = values[:, take]
        good = np.isfinite(part).mean(axis=1) >= 0.8
        out[good, j] = np.nanmean(part[good], axis=1)
    return out


def climate_by_station(years: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    days = pd.date_range("2014-01-01", "2025-12-31", freq="D")
    stations = pd.read_csv(CLIM / "stations.csv")
    temp_codes = stations.loc[stations["temp"] == 1, "code"].tolist()
    temp, temp_kept = station_matrix(temp_codes, "TEMP", days)
    lookup = {code: i for i, code in enumerate(temp_kept)}
    dew_codes = [code for code in temp_kept if int(stations.set_index("code").loc[code, "dew"]) == 1]
    dew, dew_kept = station_matrix(dew_codes, "DEW", days)
    dew_lookup = {code: i for i, code in enumerate(dew_kept)}
    vpd_rows = []
    vpd_codes = []
    for code in dew_kept:
        if code not in lookup:
            continue
        t = temp[lookup[code]]
        d = dew[dew_lookup[code]]
        both = np.isfinite(t) & np.isfinite(d)
        row = np.full(t.shape, np.nan)
        row[both] = np.clip(es_kpa(t[both]) - es_kpa(d[both]), 0, None)
        vpd_rows.append(row)
        vpd_codes.append(code)
    vpd = np.vstack(vpd_rows)
    meta = stations.set_index("code")
    temp_xy = local_km(meta.loc[temp_kept, "lon"].to_numpy(np.float64), meta.loc[temp_kept, "lat"].to_numpy(np.float64))
    vpd_xy = local_km(meta.loc[vpd_codes, "lon"].to_numpy(np.float64), meta.loc[vpd_codes, "lat"].to_numpy(np.float64))
    return {
        "temp_jja": (season_station(temp, days, years, (6, 7, 8)), temp_xy),
        "temp_son": (season_station(temp, days, years, (9, 10, 11)), temp_xy),
        "vpd_mam": (season_station(vpd, days, years, (3, 4, 5)), vpd_xy),
    }


def load_dates() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(PHENO) as src:
        years = np.array(src.variables["year"][:], np.int32)
        row = np.array(src.variables["row"][:], np.int32)
        col = np.array(src.variables["col"][:], np.int32)
        sos = np.array(src.variables["sos15"][:], np.float32)
        eos = np.array(src.variables["eos50"][:], np.float32)
    short = np.isfinite(sos) & np.isfinite(eos) & (eos <= sos + 30.0)
    sos[short] = np.nan
    eos[short] = np.nan
    return years, row, col, sos, eos


def grid_series(row, col, sos, eos, grid_m: int, climate: dict) -> dict[str, np.ndarray]:
    pix = int(round(grid_m / PIX))
    key = (row // pix) * 100000 + (col // pix)
    uniq, inv = np.unique(key, return_inverse=True)
    n_cells = uniq.size
    sos_mean, _, sos_n = aggregate_count(sos, inv, n_cells)
    eos_mean, _, eos_n = aggregate_count(eos, inv, n_cells)
    cr, cc = uniq // 100000, uniq % 100000
    x = X0 + (cc * pix + (pix - 1) / 2) * PIX
    y = Y0 - (cr * pix + (pix - 1) / 2) * PIX
    lon, lat = TO_LL.transform(x, y)
    cell_xy = local_km(np.asarray(lon, np.float64), np.asarray(lat, np.float64))
    out = {"sos": sos_mean, "eos": eos_mean, "sos_n": sos_n, "eos_n": eos_n}
    for name, (station_years, station_xy) in climate.items():
        neighbor, weight = idw_weights(cell_xy, station_xy)
        out[name] = interpolate(station_years, neighbor, weight)
    return out


def aggregate_count(values: np.ndarray, inv: np.ndarray, n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_years = values.shape[1]
    mean = np.full((n_cells, n_years), np.nan, np.float32)
    count = np.zeros((n_cells, n_years), np.int32)
    for year in range(n_years):
        column = values[:, year]
        ok = np.isfinite(column)
        if not np.any(ok):
            continue
        cnt = np.bincount(inv[ok], minlength=n_cells)
        total = np.bincount(inv[ok], weights=column[ok].astype(np.float64), minlength=n_cells)
        keep = cnt >= MIN_PIXELS
        mean[keep, year] = (total[keep] / cnt[keep]).astype(np.float32)
        count[:, year] = cnt
    return mean, mean, count


def paired(y: np.ndarray, x: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y) & np.isfinite(x) & (count >= MIN_PIXELS)
    return y[mask], x[mask]


def anomaly_pair(y: np.ndarray, x: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    both = np.isfinite(y) & np.isfinite(x) & (count >= MIN_PIXELS)
    n = both.sum(axis=1)
    keep = n >= MIN_YEARS
    y_sum = np.where(both, y, 0.0).sum(axis=1)
    x_sum = np.where(both, x, 0.0).sum(axis=1)
    y_mean = np.divide(y_sum, n, out=np.zeros(n.shape), where=n > 0)
    x_mean = np.divide(x_sum, n, out=np.zeros(n.shape), where=n > 0)
    use = both & keep[:, None]
    return (y - y_mean[:, None])[use], (x - x_mean[:, None])[use]


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    if x.size < 6 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan, np.nan, int(x.size)
    r, p = pearsonr(x, y)
    return float(r), float(p), int(x.size)


def draw(bundles: dict, path: Path) -> None:
    from plot_style import annotate_years, arial, use_arial
    use_arial()
    font = arial(12)
    letter_font = arial(14)
    columns = [("hk", "Hong Kong")] + [(grid, f"{grid // 1000} km") for grid in GRIDS]
    ylabels = ("Green-up (day)", "Dormancy (day)", "Dormancy (day)")
    xlabels = ("Spring VPD (kPa)", "Autumn temperature (°C)", "Summer temperature (°C)")
    captions = (
        "(a)  Green-up and spring VPD",
        "(b)  Dormancy and autumn temperature",
        "(c)  Dormancy and summer temperature",
    )
    fig, axes = plt.subplots(len(PAIRS), len(columns), figsize=(16.4, 10.2), dpi=130)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.96, bottom=0.12, hspace=0.48, wspace=0.32)
    year_list = None
    for row, (xname, yname, title, xlabel, ylabel) in enumerate(PAIRS):
        del title, xlabel, ylabel
        for col, (key, label) in enumerate(columns):
            ax = axes[row, col]
            pack = bundles[(key, xname, yname)]
            years = pack["year"]
            x = pack["x"]
            y = pack["y"]
            year_list = sorted(np.unique(years))
            cmap = plt.colormaps["viridis"].resampled(len(year_list))
            for i, year in enumerate(year_list):
                take = years == year
                ax.scatter(
                    x[take], y[take],
                    s=42 if key == "hk" else 13,
                    color=cmap(i),
                    alpha=1.0 if key == "hk" else 0.82,
                    linewidths=0.6 if key == "hk" else 0,
                    edgecolors="white" if key == "hk" else "none",
                    zorder=4 if key == "hk" else 2,
                    label=str(int(year)) if row == 0 and col == 0 else None,
                )
            if x.size >= 2 and np.std(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                xs = np.linspace(np.min(x), np.max(x), 40)
                ax.plot(xs, intercept + slope * xs, color="#222222", lw=1.35, zorder=3)
            if key == "hk":
                annotate_years(ax, years, x, y, size=8, inside=True)
            ax.set_title(f"{label}   r = {pack['r']:+.2f}", fontproperties=font, fontsize=12, color="black", pad=3)
            ax.set_xlabel(xlabels[row], fontproperties=font, fontsize=12, color="black")
            if col == 0:
                ax.set_ylabel(ylabels[row], fontproperties=font, fontsize=12, color="black")
                ax.text(
                    0.0, 1.14, captions[row],
                    transform=ax.transAxes,
                    ha="left", va="bottom",
                    fontproperties=letter_font,
                    color="black",
                    clip_on=False,
                )
            ax.tick_params(labelsize=10, colors="black")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=11, frameon=False, prop=arial(11), bbox_to_anchor=(0.5, 0.02))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    years, row, col, sos, eos = load_dates()
    print(f"pixels {sos.shape[0]} years {years.tolist()}", flush=True)
    climate = climate_by_station(years)
    hk = pd.read_csv(FIG / "driver_scatter_points.csv")
    bundles = {}
    rows = []
    hk_map = {
        ("vpd_mam", "sos"): (hk["vpd_mam"].to_numpy(), hk["sos"].to_numpy()),
        ("temp_son", "eos"): (hk["temp_son"].to_numpy(), hk["eos"].to_numpy()),
        ("temp_jja", "eos"): (hk["temp_jja"].to_numpy(), hk["eos"].to_numpy()),
    }
    for xname, yname, title, *_ in PAIRS:
        x, y = hk_map[(xname, yname)]
        r, p, n = corr(x, y)
        bundles[("hk", xname, yname)] = {"x": x, "y": y, "year": hk["year"].to_numpy(), "r": r, "n": n, "anomaly_r": np.nan}
        rows.append({"grid_m": 0, "pair": title, "r": r, "p": p, "n": n, "anomaly_r": np.nan, "anomaly_n": 0, "median_cell_r": np.nan, "frac_positive": np.nan, "n_cells": 1})
    for grid in GRIDS:
        print(f"grid {grid}", flush=True)
        series = grid_series(row, col, sos, eos, grid, climate)
        for xname, yname, title, *_ in PAIRS:
            count = series["sos_n"] if yname == "sos" else series["eos_n"]
            y = series[yname]
            x = series[xname]
            xv, yv = paired(x, y, count)
            r, p, n = corr(xv, yv)
            axv, ayv = anomaly_pair(x, y, count)
            ar, ap, an = corr(axv, ayv)
            masked_y = np.where(count >= MIN_PIXELS, y, np.nan)
            masked_x = np.where(count >= MIN_PIXELS, x, np.nan)
            rs = cell_r(masked_y, masked_x)
            bundles[(grid, xname, yname)] = {
                "x": xv, "y": yv,
                "year": year_labels(years, x, y, count),
                "r": r, "n": n, "anomaly_r": ar,
            }
            rows.append({
                "grid_m": grid,
                "pair": title,
                "r": r,
                "p": p,
                "n": n,
                "anomaly_r": ar,
                "anomaly_p": ap,
                "anomaly_n": an,
                "median_cell_r": float(np.median(rs)) if rs.size else np.nan,
                "frac_positive": float(np.mean(rs > 0)) if rs.size else np.nan,
                "n_cells": int(rs.size),
            })
            print(f"  {title}: pooled r {r:+.3f} n {n}  anomaly r {ar:+.3f}  median cell r {np.median(rs):+.3f} cells {rs.size}", flush=True)
    table = pd.DataFrame(rows)
    table_path = FIG / "driver_grid_scale_correlations.csv"
    table.to_csv(table_path, index=False)
    final = FIG / "final" / "07_driver_grid_scale_scatters.png"
    draw(bundles, final)
    print(f"wrote {final}")
    print(f"wrote {table_path}")


def year_labels(years: np.ndarray, x: np.ndarray, y: np.ndarray, count: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x) & np.isfinite(y) & (count >= MIN_PIXELS)
    # mask is (cells, years); ravel is C order, year varies slowest if we transpose to (years, cells)
    return np.broadcast_to(years, x.shape)[mask]


if __name__ == "__main__":
    main()
