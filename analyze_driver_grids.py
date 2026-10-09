#!/usr/bin/env python3
"""SOS (15%) and EOS (50%) against HKO climate, including VPD.

Phenology stays on the 30 m pixels, then each analysis cell summarizes the
pixels inside it (mean and spread). Climate is inverse-distance weighted from
stations onto the cell center, not onto every 30 m pixel. Station spacing is
several kilometres, so the grids are 2, 5 and 10 km.

VPD is es(temperature) - es(dew point), Tetens, in kPa, computed at each
station before interpolation.

Two year sets: 2015–2025, and 2016–2025 with the first HLS year left out.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from netCDF4 import Dataset
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent
PHENO = Path(os.environ.get("HK_PHENO_ROOT", ROOT / "data/phenology"))
SRC = PHENO / "phenology_threshold_sl122_evi.nc"
CLIM = PHENO / "climate"
FIG = ROOT / "figures"
GRIDS_M = (2000, 5000, 10000)
MIN_PIXELS = 80
MIN_YEARS = 7
X0, Y0, PIX = 799980.0, 2500020.0, 30.0
TO_LL = pyproj.Transformer.from_crs(32649, 4326, always_xy=True)


def es_kpa(temp_c: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def local_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    x = (lon - 114.1) * 111.32 * np.cos(np.deg2rad(22.3))
    y = (lat - 22.3) * 110.57
    return np.column_stack([x, y])


def load_daily(path: Path, index: pd.DatetimeIndex) -> pd.Series:
    raw = pd.read_csv(path, skiprows=3, encoding="utf-8-sig")
    raw.columns = ["year", "month", "day", "hour", "minute", "second", "tz", "value", "flag"]
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    when = pd.to_datetime(dict(year=raw["year"], month=raw["month"], day=raw["day"]), errors="coerce")
    series = pd.Series(raw["value"].to_numpy(np.float64), index=when).groupby(level=0).mean()
    return series.reindex(index)


def station_matrix(codes: list[str], suffix: str, index: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    columns = []
    kept = []
    for code in codes:
        path = CLIM / f"daily_{code}_{suffix}_ALL.csv"
        if not path.exists():
            continue
        columns.append(load_daily(path, index).to_numpy(np.float64))
        kept.append(code)
    if not columns:
        raise RuntimeError(f"no files for {suffix}")
    return np.column_stack(columns).T, kept


def idw_weights(cell_xy: np.ndarray, station_xy: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    dist = cdist(cell_xy, station_xy)
    dist = np.maximum(dist, 0.05)
    k = min(k, station_xy.shape[0])
    take = np.argpartition(dist, k - 1, axis=1)[:, :k]
    chosen = np.take_along_axis(dist, take, axis=1)
    order = np.argsort(chosen, axis=1)
    take = np.take_along_axis(take, order, axis=1)
    chosen = np.take_along_axis(chosen, order, axis=1)
    weight = 1.0 / chosen ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return take.astype(np.int32), weight


def interpolate(values: np.ndarray, neighbor: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """values (stations, days) -> (cells, days), renormalized over finite stations."""
    picked = values[neighbor]  # cells, k, days
    valid = np.isfinite(picked)
    w = np.where(valid, weight[:, :, None], 0.0)
    denom = w.sum(axis=1)
    num = np.nansum(np.where(valid, picked, 0.0) * w, axis=1)
    out = np.divide(num, denom, out=np.full(num.shape, np.nan), where=denom > 0)
    return out.astype(np.float64)


def window_stat(daily: np.ndarray, end: np.ndarray, length: int, as_sum: bool, min_frac: float = 0.8) -> np.ndarray:
    """daily (cells, days), end (cells,) inclusive index. Returns (cells,)."""
    n, ndays = daily.shape
    csum = np.cumsum(np.where(np.isfinite(daily), daily, 0.0), axis=1)
    ccnt = np.cumsum(np.isfinite(daily).astype(np.int32), axis=1)
    start = end - length + 1
    out = np.full(n, np.nan)
    ok = (end >= 0) & (start >= 0) & (end < ndays)
    if not np.any(ok):
        return out
    idx = np.arange(n)
    end_ok = end[ok]
    start_ok = start[ok]
    total = csum[idx[ok], end_ok] - np.where(start_ok > 0, csum[idx[ok], start_ok - 1], 0.0)
    count = ccnt[idx[ok], end_ok] - np.where(start_ok > 0, ccnt[idx[ok], start_ok - 1], 0)
    good = count >= min_frac * length
    stat = total if as_sum else total / np.maximum(count, 1)
    out[idx[ok][good]] = stat[good]
    return out


def nearest_neighbor_km(xy: np.ndarray) -> float:
    dist = cdist(xy, xy)
    np.fill_diagonal(dist, np.inf)
    return float(np.median(dist.min(axis=1)))


def aggregate(values: np.ndarray, inv: np.ndarray, n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """values (pixels, years) -> mean, sample std, count, each (cells, years)."""
    n_years = values.shape[1]
    mean = np.full((n_cells, n_years), np.nan)
    std = np.full((n_cells, n_years), np.nan)
    count = np.zeros((n_cells, n_years), np.int32)
    for year in range(n_years):
        column = values[:, year]
        ok = np.isfinite(column)
        if not np.any(ok):
            continue
        cnt = np.bincount(inv[ok], minlength=n_cells).astype(np.float64)
        total = np.bincount(inv[ok], weights=column[ok], minlength=n_cells)
        sq = np.bincount(inv[ok], weights=column[ok] ** 2, minlength=n_cells)
        keep = cnt >= MIN_PIXELS
        mu = np.full(n_cells, np.nan)
        mu[keep] = total[keep] / cnt[keep]
        var = np.full(n_cells, np.nan)
        # sample variance
        var[keep] = (sq[keep] - cnt[keep] * mu[keep] ** 2) / np.maximum(cnt[keep] - 1.0, 1.0)
        var[var < 0] = 0
        mean[:, year] = mu
        std[:, year] = np.sqrt(var)
        count[:, year] = cnt.astype(np.int32)
    return mean, std, count


def cell_r(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    rs = []
    for i in range(y.shape[0]):
        mask = np.isfinite(y[i]) & np.isfinite(x[i])
        if int(mask.sum()) < MIN_YEARS:
            continue
        if np.std(y[i, mask]) < 1e-6 or np.std(x[i, mask]) < 1e-6:
            continue
        r, _ = pearsonr(y[i, mask], x[i, mask])
        rs.append(r)
    return np.asarray(rs, np.float64)


def territory_r(y: np.ndarray, x: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(y) & np.isfinite(x)
    n = int(mask.sum())
    if n < 6 or np.std(y[mask]) < 1e-6 or np.std(x[mask]) < 1e-6:
        return np.nan, np.nan, n
    r, p = pearsonr(y[mask], x[mask])
    return float(r), float(p), n


def summarize(rs: np.ndarray) -> dict[str, float]:
    if rs.size == 0:
        return {"median_r": np.nan, "p25": np.nan, "p75": np.nan, "n_cells": 0, "frac_positive": np.nan}
    return {
        "median_r": float(np.median(rs)),
        "p25": float(np.percentile(rs, 25)),
        "p75": float(np.percentile(rs, 75)),
        "n_cells": int(rs.size),
        "frac_positive": float(np.mean(rs > 0)),
    }


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    days = pd.date_range("2014-01-01", "2025-12-31", freq="D")
    stations = pd.read_csv(CLIM / "stations.csv")
    with Dataset(SRC) as src:
        years = np.array(src.variables["year"][:], np.int32)
        row = np.array(src.variables["row"][:], np.int32)
        col = np.array(src.variables["col"][:], np.int32)
        sos = np.array(src.variables["sos15"][:], np.float64)
        eos = np.array(src.variables["eos50"][:], np.float64)
    short = np.isfinite(sos) & np.isfinite(eos) & (eos <= sos + 30.0)
    sos[short] = np.nan
    eos[short] = np.nan
    print(f"years {years.tolist()} pixels {sos.shape[0]}", flush=True)

    # City-wide retrieval, independent of the grid.
    yearly_rows = []
    for i, year in enumerate(years):
        yearly_rows.append({
            "year": int(year),
            "sos_median": float(np.nanmedian(sos[:, i])),
            "eos_median": float(np.nanmedian(eos[:, i])),
            "sos_valid": float(np.isfinite(sos[:, i]).mean()),
            "eos_valid": float(np.isfinite(eos[:, i]).mean()),
        })
    yearly = pd.DataFrame(yearly_rows)
    print(yearly.to_string(index=False), flush=True)

    networks = {}
    for label, flag, suffix in (
        ("temp", "temp", "TEMP"),
        ("rain", "rain", "RF"),
        ("wind", "wind", "WSPD"),
    ):
        codes = stations.loc[stations[flag] == 1, "code"].tolist()
        values, kept = station_matrix(codes, suffix, days)
        meta = stations.set_index("code").loc[kept]
        xy = local_km(meta["lon"].to_numpy(np.float64), meta["lat"].to_numpy(np.float64))
        networks[label] = {"values": values, "xy": xy, "codes": kept, "sum": label == "rain"}
        print(f"{label} stations {len(kept)} median spacing {nearest_neighbor_km(xy):.2f} km", flush=True)

    temp_meta = stations.set_index("code").loc[networks["temp"]["codes"]]
    dew_codes = [c for c in networks["temp"]["codes"] if int(stations.set_index("code").loc[c, "dew"]) == 1]
    dew_values, dew_kept = station_matrix(dew_codes, "DEW", days)
    # Align dew to the temperature station order used above.
    order = {code: i for i, code in enumerate(dew_kept)}
    temp_for_vpd = []
    dew_for_vpd = []
    vpd_codes = []
    temp_lookup = {code: i for i, code in enumerate(networks["temp"]["codes"])}
    for code in dew_kept:
        if code not in temp_lookup:
            continue
        t = networks["temp"]["values"][temp_lookup[code]]
        d = dew_values[order[code]]
        both = np.isfinite(t) & np.isfinite(d)
        vpd = np.full(t.shape, np.nan)
        vpd[both] = np.clip(es_kpa(t[both]) - es_kpa(d[both]), 0, None)
        temp_for_vpd.append(vpd)
        vpd_codes.append(code)
    vpd_meta = stations.set_index("code").loc[vpd_codes]
    vpd_xy = local_km(vpd_meta["lon"].to_numpy(np.float64), vpd_meta["lat"].to_numpy(np.float64))
    networks["vpd"] = {
        "values": np.vstack(temp_for_vpd),
        "xy": vpd_xy,
        "codes": vpd_codes,
        "sum": False,
    }
    print(f"vpd stations {len(vpd_codes)} median spacing {nearest_neighbor_km(vpd_xy):.2f} km", flush=True)
    spacing = nearest_neighbor_km(networks["temp"]["xy"])
    primary = min(GRIDS_M, key=lambda g: abs(g / 1000 - spacing))
    print(f"primary grid {primary} m, nearest the {spacing:.2f} km temperature spacing", flush=True)

    # Territory climate: mean across stations, one Hong Kong series.
    territory_daily = {name: np.nanmean(net["values"], axis=0) for name, net in networks.items()}

    def slice_1d(series: np.ndarray, end: int, length: int, as_sum: bool) -> float:
        start = end - length + 1
        if start < 0 or end >= series.size:
            return np.nan
        part = series[start : end + 1]
        if np.isfinite(part).mean() < 0.8:
            return np.nan
        return float(np.nansum(part) if as_sum else np.nanmean(part))

    def season_bounds(year: int, name: str) -> tuple[int, int]:
        if name == "DJF":
            a, b = pd.Timestamp(year - 1, 12, 1), pd.Timestamp(year, 2, 28)
        elif name == "MAM":
            a, b = pd.Timestamp(year, 3, 1), pd.Timestamp(year, 5, 31)
        elif name == "JJA":
            a, b = pd.Timestamp(year, 6, 1), pd.Timestamp(year, 8, 31)
        else:
            a, b = pd.Timestamp(year, 9, 1), pd.Timestamp(year, 11, 30)
        return int((a - days[0]).days), int((b - days[0]).days)

    records = []
    versions = {
        "2015-2025": np.ones(len(years), dtype=bool),
        "2016-2025": years >= 2016,
    }
    # Precompute territory windows against the city median date.
    city = {}
    for response, target in (("SOS", sos), ("EOS", eos)):
        med = np.array([np.nanmedian(target[:, i]) for i in range(len(years))])
        city[response] = med
        for label, net in networks.items():
            series = territory_daily[label]
            for lag in (30, 60, 90):
                vals = []
                for i, year in enumerate(years):
                    end = int((pd.Timestamp(int(year), 1, 1) - days[0]).days + int(round(med[i])) - 1)
                    vals.append(slice_1d(series, end, lag, net["sum"]))
                city[(response, label, f"{lag}d")] = np.asarray(vals, np.float64)
            for season in (("DJF", "MAM") if response == "SOS" else ("JJA", "SON")):
                vals = []
                for year in years:
                    a, b = season_bounds(int(year), season)
                    length = b - a + 1
                    vals.append(slice_1d(series, b, length, net["sum"]))
                city[(response, label, season)] = np.asarray(vals, np.float64)

    spread_by_grid = {}
    for grid_m in GRIDS_M:
        pix = int(round(grid_m / PIX))
        key = (row // pix) * 10000 + (col // pix)
        uniq, inv = np.unique(key, return_inverse=True)
        n_cells = uniq.size
        sos_mean, sos_std, sos_n = aggregate(sos, inv, n_cells)
        eos_mean, eos_std, eos_n = aggregate(eos, inv, n_cells)
        cr, cc = uniq // 10000, uniq % 10000
        row_c = cr * pix + (pix - 1) / 2
        col_c = cc * pix + (pix - 1) / 2
        x = X0 + col_c * PIX
        y = Y0 - row_c * PIX
        lon, lat = TO_LL.transform(x, y)
        cell_xy = local_km(np.asarray(lon, np.float64), np.asarray(lat, np.float64))
        active = (sos_n >= MIN_PIXELS).any(axis=1)
        print(
            f"grid {grid_m} m cells {int(active.sum())} with >= {MIN_PIXELS} pixels in some year",
            flush=True,
        )
        daily_cells = {}
        for label, net in networks.items():
            neighbor, weight = idw_weights(cell_xy, net["xy"])
            daily_cells[label] = interpolate(net["values"], neighbor, weight)

        spread_by_grid[grid_m] = {
            "year": years,
            "sos_std_median": np.nanmedian(np.where(sos_n >= MIN_PIXELS, sos_std, np.nan), axis=0),
            "eos_std_median": np.nanmedian(np.where(eos_n >= MIN_PIXELS, eos_std, np.nan), axis=0),
        }

        responses = {
            "sos_mean": sos_mean,
            "sos_std": sos_std,
            "eos_mean": eos_mean,
            "eos_std": eos_std,
        }
        windows = {
            "SOS": [("30d", 30), ("60d", 60), ("90d", 90), ("DJF", None), ("MAM", None)],
            "EOS": [("30d", 30), ("60d", 60), ("90d", 90), ("JJA", None), ("SON", None)],
        }
        for response_name, response in responses.items():
            kind = "SOS" if response_name.startswith("sos") else "EOS"
            for label, net in networks.items():
                daily = daily_cells[label]
                for window_name, lag in windows[kind]:
                    driver = np.full(response.shape, np.nan)
                    for i, year in enumerate(years):
                        if lag is not None:
                            doy = sos_mean[:, i] if kind == "SOS" else eos_mean[:, i]
                            end = np.full(n_cells, -1, np.int32)
                            finite = np.isfinite(doy)
                            end[finite] = (
                                int((pd.Timestamp(int(year), 1, 1) - days[0]).days)
                                + np.rint(doy[finite]).astype(np.int32)
                                - 1
                            )
                            driver[:, i] = window_stat(daily, end, lag, net["sum"])
                        else:
                            a, b = season_bounds(int(year), window_name)
                            end = np.full(n_cells, b, np.int32)
                            driver[:, i] = window_stat(daily, end, b - a + 1, net["sum"])
                    for version, keep_year in versions.items():
                        rs = cell_r(response[:, keep_year], driver[:, keep_year])
                        stats = summarize(rs)
                        # Territory number uses the city median date and the station mean.
                        if response_name.endswith("mean"):
                            terr_y = city[kind][keep_year]
                            terr_x = city[(kind, label, window_name)][keep_year]
                            tr, tp, tn = territory_r(terr_y, terr_x)
                        else:
                            # Median within-cell spread that year, same climate series.
                            spread = np.nanmedian(
                                np.where(
                                    (sos_n if kind == "SOS" else eos_n) >= MIN_PIXELS,
                                    response,
                                    np.nan,
                                ),
                                axis=0,
                            )
                            terr_x = city[(kind, label, window_name)][keep_year]
                            tr, tp, tn = territory_r(spread[keep_year], terr_x)
                        records.append({
                            "grid_m": grid_m,
                            "version": version,
                            "response": response_name,
                            "driver": label,
                            "window": window_name,
                            **stats,
                            "territory_r": tr,
                            "territory_p": tp,
                            "territory_n": tn,
                        })
        print(f"grid {grid_m} done", flush=True)

    table = pd.DataFrame(records)
    table_path = FIG / "driver_lag_correlations.csv"
    table.to_csv(table_path, index=False)
    yearly.to_csv(FIG / "phenology_yearly_sos15_eos50.csv", index=False)
    print(f"wrote {table_path}", flush=True)

    # Figure for the grid closest to the station spacing.
    sub = table[(table["grid_m"] == primary) & table["response"].isin(["sos_mean", "eos_mean"])]
    drivers = ["temp", "rain", "wind", "vpd"]
    driver_label = {"temp": "Temperature", "rain": "Rainfall", "wind": "Wind", "vpd": "VPD"}
    panels = [
        ("sos_mean", "2015-2025", "SOS, 2015–2025", ["30d", "60d", "90d", "DJF", "MAM"]),
        ("sos_mean", "2016-2025", "SOS, 2016–2025", ["30d", "60d", "90d", "DJF", "MAM"]),
        ("eos_mean", "2015-2025", "EOS, 2015–2025", ["30d", "60d", "90d", "JJA", "SON"]),
        ("eos_mean", "2016-2025", "EOS, 2016–2025", ["30d", "60d", "90d", "JJA", "SON"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.6), sharey=True)
    for ax, (response, version, title, cols) in zip(axes.ravel(), panels):
        block = sub[(sub["response"] == response) & (sub["version"] == version)]
        mat = np.full((len(drivers), len(cols)), np.nan)
        for i, driver in enumerate(drivers):
            for j, window in enumerate(cols):
                hit = block[(block["driver"] == driver) & (block["window"] == window)]
                if len(hit):
                    mat[i, j] = hit["median_r"].iloc[0]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(len(drivers)), [driver_label[d] for d in drivers])
        ax.set_title(title, loc="left", fontsize=11)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label="Median correlation across cells")
    fig.suptitle(
        f"Year-to-year correlation of cell-mean phenology with pre-season climate\n"
        f"{primary // 1000} km cells, matched to the {spacing:.1f} km temperature-station spacing. "
        f"SOS is 15% of green-up, EOS is 50% of senescence.",
        fontsize=12,
    )
    fig.savefig(FIG / "driver_lag_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Yearly phenology and within-cell spread, so 2015 can be compared.
    spread = spread_by_grid[primary]
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.2), sharex=True)
    axes[0].plot(yearly["year"], yearly["sos_median"], color="#1b7f4e", marker="o", label="SOS, 15%")
    axes[0].plot(yearly["year"], yearly["eos_median"], color="#b86e00", marker="o", label="EOS, 50%")
    axes[0].axvline(2015, color="#888888", lw=0.8, ls="--")
    axes[0].set_ylabel("Median day of year")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].set_title("Hong Kong median dates, and how widely dates scatter inside a cell", loc="left")
    axes[1].plot(spread["year"], spread["sos_std_median"], color="#1b7f4e", marker="o", label="SOS spread")
    axes[1].plot(spread["year"], spread["eos_std_median"], color="#b86e00", marker="o", label="EOS spread")
    axes[1].axvline(2015, color="#888888", lw=0.8, ls="--")
    axes[1].set_ylabel(f"Median within-cell standard deviation, {primary // 1000} km")
    axes[1].set_xlabel("Year")
    axes[1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "phenology_yearly_sos15_eos50.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures", flush=True)

    # Print the territory correlations for the reply.
    show = table[(table["grid_m"] == primary) & table["response"].isin(["sos_mean", "eos_mean"])]
    cols = ["version", "response", "driver", "window", "territory_r", "territory_p", "territory_n", "median_r", "n_cells", "frac_positive"]
    print(show[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
