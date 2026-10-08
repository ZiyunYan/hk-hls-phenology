#!/usr/bin/env python3
"""Phenology versus Hong Kong Observatory climate and distance to built-up.

Spatial position stays on the pixel. Interannual effects use anomalies, so a
pixel's own mean is removed. Lags are the 30, 60 and 90 days before SOS or EOS,
plus fixed seasons: DJF and MAM before SOS, JJA and SON before EOS.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from numba import njit
from scipy.stats import pearsonr

ROOT = Path("/intelnvme03/ziyun218/hls_49QHE_hk/phenology")
PHENO = ROOT / "phenology_sos_eos.nc"
CLIM = ROOT / "climate"
OUT = ROOT / "analysis"
YEAR0 = 2015
N_YEARS = 11
LAGS = (30, 60, 90)
CLASS_NAME = {61: "agriculture", 71: "woodland", 72: "shrubland", 73: "grassland", 74: "mangrove"}


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def load_daily(paths: list[Path], day_index: pd.DatetimeIndex) -> np.ndarray:
    frame = np.full((len(paths), len(day_index)), np.nan, np.float32)
    for i, path in enumerate(paths):
        raw = pd.read_csv(path, skiprows=3, encoding="utf-8-sig")
        raw.columns = ["year", "month", "day", "hour", "minute", "second", "tz", "value", "flag"]
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        when = pd.to_datetime(dict(year=raw["year"], month=raw["month"], day=raw["day"]), errors="coerce")
        series = pd.Series(raw["value"].to_numpy(), index=when).groupby(level=0).mean()
        frame[i] = series.reindex(day_index).to_numpy(np.float32)
    return frame


def local_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    x = (lon - 114.1) * 111.32 * np.cos(np.deg2rad(22.3))
    y = (lat - 22.3) * 110.57
    return np.column_stack([x, y]).astype(np.float64)


def idw(pixel_xy: np.ndarray, station_xy: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    # chunked distances so 800k x stations stays small
    n = pixel_xy.shape[0]
    idx = np.empty((n, k), np.int32)
    w = np.empty((n, k), np.float64)
    for p0 in range(0, n, 20000):
        p1 = min(p0 + 20000, n)
        d = pixel_xy[p0:p1, None, :] - station_xy[None, :, :]
        dist = np.sqrt((d * d).sum(axis=2))
        dist = np.maximum(dist, 0.05)
        take = np.argpartition(dist, k, axis=1)[:, :k]
        chosen = np.take_along_axis(dist, take, axis=1)
        order = np.argsort(chosen, axis=1)
        take = np.take_along_axis(take, order, axis=1)
        chosen = np.take_along_axis(chosen, order, axis=1)
        weight = 1.0 / chosen ** 2
        weight /= weight.sum(axis=1, keepdims=True)
        idx[p0:p1] = take
        w[p0:p1] = weight
    return idx, w


@njit
def preseason(values, neighbor, weight, sos, year_ord0, out, as_sum):
    """values: (stations, days). sos: (pixels, years) DOY. out: (pixels, years, 3 lags)."""
    n, nyear = sos.shape
    ndays = values.shape[1]
    for i in range(n):
        for y in range(nyear):
            doy = sos[i, y]
            if not np.isfinite(doy):
                continue
            end = year_ord0[y] + int(doy) - 1
            for lag_i in range(3):
                lag = 30 * (lag_i + 1)
                start = end - lag + 1
                if start < 0 or end >= ndays:
                    continue
                acc = 0.0
                wt = 0.0
                for nb in range(neighbor.shape[1]):
                    s = neighbor[i, nb]
                    part = 0.0
                    count = 0
                    for d in range(start, end + 1):
                        v = values[s, d]
                        if np.isfinite(v):
                            part += v
                            count += 1
                    if count == 0:
                        continue
                    if as_sum == 0:
                        part = part / count
                    acc += weight[i, nb] * part
                    wt += weight[i, nb]
                if wt > 0:
                    out[i, y, lag_i] = acc / wt


def corr(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < 30:
        return np.nan, n
    r, _ = pearsonr(a[m], b[m])
    return float(r), n


def anomaly(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=1, keepdims=True)
    return x - mu


@njit
def pixel_slopes(y):
    n, t = y.shape
    out = np.empty(n)
    for i in range(n):
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        m = 0
        for k in range(t):
            if not np.isfinite(y[i, k]):
                continue
            sx += k
            sy += y[i, k]
            sxx += k * k
            sxy += k * y[i, k]
            m += 1
        if m < 6:
            out[i] = np.nan
            continue
        den = m * sxx - sx * sx
        out[i] = (m * sxy - sx * sy) / den if den != 0 else np.nan
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(CLIM / "stations.csv")
    with Dataset(PHENO) as f:
        sos = np.array(f.variables["sos"][:], np.float64)
        eos = np.array(f.variables["eos"][:], np.float64)
        lon = np.array(f.variables["lon"][:], np.float64)
        lat = np.array(f.variables["lat"][:], np.float64)
        lum = np.array(f.variables["lum_code"][:], np.int16)
        dist = np.array(f.variables["dist_urban_m"][:], np.float64)
        years = np.array(f.variables["year"][:], np.int16)
    log(f"pixels {sos.shape[0]}")

    days = pd.date_range("2014-01-01", "2025-12-31", freq="D")
    year_ord0 = np.array([(pd.Timestamp(int(y), 1, 1) - days[0]).days for y in years], np.int32)

    def series_for(flag: str, suffix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        use = stations[stations[flag] == 1]
        paths = [CLIM / f"daily_{code}_{suffix}_ALL.csv" for code in use["code"]]
        paths = [p for p in paths if p.exists()]
        codes = [p.name.split("_")[1] for p in paths]
        meta = use.set_index("code").loc[codes]
        values = load_daily(paths, days)
        idx, w = idw(local_km(lon, lat), local_km(meta["lon"].to_numpy(), meta["lat"].to_numpy()))
        return values, idx, w

    rows = []
    for label, target, flag, suffix, how in (
        ("temp", sos, "temp", "TEMP", "mean"),
        ("rain", sos, "rain", "RF", "sum"),
        ("wind", sos, "wind", "WSPD", "mean"),
    ):
        log(f"lag {label} vs SOS")
        values, neighbor, weight = series_for(flag, suffix)
        # rainfall preseason should be a sum; scale the mean by lag length below
        out = np.full((sos.shape[0], N_YEARS, 3), np.nan, np.float64)
        preseason(values, neighbor, weight, target, year_ord0, out, 1 if how == "sum" else 0)
        target_a = anomaly(target)
        for lag_i, lag in enumerate(LAGS):
            driver_a = anomaly(out[:, :, lag_i])
            r, n = corr(target_a.ravel(), driver_a.ravel())
            rows.append({"response": "SOS", "driver": label, "lag_days": lag, "r_anomaly": r, "n": n})
            log(f"  SOS {label} {lag}d r={r:.3f} n={n}")

    # EOS uses the same three drivers
    for label, flag, suffix, how in (
        ("temp", "temp", "TEMP", "mean"),
        ("rain", "rain", "RF", "sum"),
        ("wind", "wind", "WSPD", "mean"),
    ):
        log(f"lag {label} vs EOS")
        values, neighbor, weight = series_for(flag, suffix)
        out = np.full((eos.shape[0], N_YEARS, 3), np.nan, np.float64)
        preseason(values, neighbor, weight, eos, year_ord0, out, 1 if how == "sum" else 0)
        target_a = anomaly(eos)
        for lag_i, lag in enumerate(LAGS):
            r, n = corr(target_a.ravel(), anomaly(out[:, :, lag_i]).ravel())
            rows.append({"response": "EOS", "driver": label, "lag_days": lag, "r_anomaly": r, "n": n})
            log(f"  EOS {label} {lag}d r={r:.3f} n={n}")

    sos_mean = np.nanmean(sos, axis=1)
    eos_mean = np.nanmean(eos, axis=1)
    factor_path = ROOT / "factors30/veg_factors.npz"
    factor_cols = {}
    if factor_path.exists():
        packed = np.load(factor_path)
        same = np.array_equal(packed["row"], np.array(Dataset(PHENO).variables["row"][:])) and np.array_equal(
            packed["col"], np.array(Dataset(PHENO).variables["col"][:])
        )
        if not same:
            log("30 m factor grid does not match phenology pixels; skipped")
        else:
            for key in packed.files:
                if key in ("row", "col"):
                    continue
                factor_cols[key] = np.array(packed[key], np.float64)
    spatial = {"dist_urban_m": dist, **factor_cols}
    for response, mu in (("SOS", sos_mean), ("EOS", eos_mean)):
        for name, values in spatial.items():
            r, n = corr(mu, values)
            rows.append({"response": response, "driver": name, "lag_days": 0, "r_anomaly": r, "n": n})
            log(f"spatial {response} vs {name} r={r:.3f}")

    yearly = []
    for y in range(N_YEARS):
        yearly.append({
            "year": int(years[y]),
            "sos_median": float(np.nanmedian(sos[:, y])),
            "eos_median": float(np.nanmedian(eos[:, y])),
            "n_sos": int(np.isfinite(sos[:, y]).sum()),
            "n_eos": int(np.isfinite(eos[:, y]).sum()),
        })
    sos_slope = pixel_slopes(sos)
    pd.DataFrame(rows).to_csv(OUT / "driver_correlations.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUT / "yearly_median_phenology.csv", index=False)
    summary = pd.DataFrame({
        "lum_code": lum,
        "class": [CLASS_NAME.get(int(c), str(c)) for c in lum],
        "lon": lon,
        "lat": lat,
        "row": np.array(Dataset(PHENO).variables["row"][:]),
        "col": np.array(Dataset(PHENO).variables["col"][:]),
        "dist_urban_m": dist,
        **{name: values for name, values in factor_cols.items()},
        "sos_mean": sos_mean,
        "eos_mean": eos_mean,
        "sos_slope_d_per_y": sos_slope,
    })
    # class medians
    summary.groupby("class")[["sos_mean", "eos_mean", "sos_slope_d_per_y", "dist_urban_m"]].median().to_csv(
        OUT / "class_median.csv"
    )
    log(f"median SOS slope {np.nanmedian(sos_slope):.3f} d/year")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
