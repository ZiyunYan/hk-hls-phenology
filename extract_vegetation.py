#!/usr/bin/env python3
"""Vegetation pixels on the Hong Kong HLS 30 m grid.

Land cover is the Planning Department Land Utilization raster (10 m, HK1980).
Each 30 m HLS cell takes the majority 10 m class. Vegetation is agricultural
land, woodland, shrubland, grassland and mangrove. Row, column, longitude and
latitude stay with every pixel.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import rasterio
from netCDF4 import Dataset
from rasterio.transform import Affine, xy
from rasterio.warp import Resampling, reproject, transform as rio_transform

from hk_paths import DATA as ROOT
LUM_TIF = ROOT / "phenology/landuse/LUMHK_RasterGrid_2024/LUM_end2024.tif"
OUT = ROOT / "phenology/veg_grid.nc"
META = json.loads((ROOT / "meta.json").read_text())

HEIGHT = 1830
WIDTH = 1830
YEAR0 = 2015
N_YEARS = 11
STEPS_PER_YEAR = 122
N_TIME = N_YEARS * STEPS_PER_YEAR
BANDS6 = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
# Native HLS slot. NIR/SWIR differ between L30 and S30.
SLOT = {"B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6, "B8A": 8, "B11": 11, "B12": 12}
VEG_CODES = (61, 71, 72, 73, 74)
# Built-up used for distance to the city. Vegetation and water are excluded.
URBAN_CODES = (1, 2, 3, 11, 21, 22, 23, 31, 41, 42, 43, 44)
CLOUDISH = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
FILL = -9999


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def hls_transform() -> Affine:
    t = META["transform"]
    return Affine(t[0], t[1], t[2], t[3], t[4], t[5])


def majority_landuse() -> np.ndarray:
    aff = hls_transform()
    with rasterio.open(LUM_TIF) as src:
        dst = np.full((HEIGHT, WIDTH), 0, np.int16)
        reproject(
            source=src.read(1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=-128,
            dst_transform=aff,
            dst_crs="EPSG:32649",
            dst_nodata=0,
            resampling=Resampling.mode,
        )
    return dst


def calendar_times() -> list[str]:
    times = []
    for year in range(YEAR0, YEAR0 + N_YEARS):
        for step in range(STEPS_PER_YEAR):
            doy = step * 3 + 1
            times.append(f"{year}{doy:03d}")
    return times


def granule_list() -> list[dict]:
    manifest = json.loads((ROOT / "granules.json").read_text())
    kept = []
    for g in manifest:
        stamp = g["datetime"][:7]
        if "2015001" <= stamp <= "2025365":
            kept.append(g)
    kept.sort(key=lambda g: (g["datetime"], g["sensor"] != "S30", g["granule_id"]))
    return kept


def bin_of(stamp: str) -> int:
    year = int(stamp[:4])
    doy = int(stamp[4:7])
    if year < YEAR0 or year >= YEAR0 + N_YEARS:
        return -1
    index = (year - YEAR0) * STEPS_PER_YEAR + (doy - 1) // 3
    if index < 0 or index >= N_TIME:
        return -1
    return index


def optical6(data: np.ndarray, sensor: str, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """data is (13, H, W) int16. Return (n, 6) float32 with cloudy pixels as NaN."""
    picked = data[:, rows, cols].astype(np.float32)
    if sensor == "L30":
        idx = [SLOT["B02"], SLOT["B03"], SLOT["B04"], SLOT["B05"], SLOT["B06"], SLOT["B07"]]
    elif sensor == "S30":
        idx = [SLOT["B02"], SLOT["B03"], SLOT["B04"], SLOT["B8A"], SLOT["B11"], SLOT["B12"]]
    else:
        raise ValueError(sensor)
    out = picked[idx].T
    return out


def main() -> None:
    log("majority-vote Planning Department land use onto the 30 m grid")
    lum = majority_landuse()
    veg = np.isin(lum, VEG_CODES)
    rows, cols = np.nonzero(veg)
    n = int(rows.size)
    log(
        f"vegetation pixels {n}: "
        + ", ".join(f"{code}={int((lum == code).sum())}" for code in VEG_CODES)
    )
    if n == 0:
        raise SystemExit("no vegetation pixels")

    aff = hls_transform()
    xs, ys = xy(aff, rows, cols, offset="center")
    lon, lat = rio_transform("EPSG:32649", "EPSG:4326", np.asarray(xs), np.asarray(ys))
    lon = np.asarray(lon, np.float32)
    lat = np.asarray(lat, np.float32)

    from scipy.ndimage import distance_transform_edt

    urban = np.isin(lum, URBAN_CODES)
    dist_m = distance_transform_edt(~urban).astype(np.float32) * 30.0
    dist_veg = dist_m[rows, cols]
    lum_veg = lum[rows, cols].astype(np.int16)

    granules = granule_list()
    bins = np.array([bin_of(g["datetime"][:7]) for g in granules], np.int32)
    keep = bins >= 0
    granules = [g for g, ok in zip(granules, keep) if ok]
    bins = bins[keep]
    ngran = len(granules)
    log(f"granules 2015-2025: {ngran}")

    series_path = ROOT / "phenology/veg_irregular.f16"
    if series_path.exists():
        series_path.unlink()
    series = np.memmap(series_path, dtype=np.float16, mode="w+", shape=(n, ngran, 6))
    series[:] = np.nan

    for gi, g in enumerate(granules):
        path = ROOT / "granules" / f"{g['granule_id']}.npz"
        with np.load(path) as z:
            fm = z["fmask"][rows, cols]
            vals = optical6(z["data"], str(z["sensor"]), rows, cols)
        bad = (fm == 255) | ((fm & CLOUDISH) != 0)
        vals[bad] = np.nan
        vals[vals == FILL] = np.nan
        incomplete = ~np.isfinite(vals).all(axis=1)
        vals[incomplete] = np.nan
        series[:, gi, :] = vals.astype(np.float16)
        if gi % 50 == 0 or gi + 1 == ngran:
            series.flush()
            log(f"read {gi + 1}/{ngran} {g['granule_id']}")

    groups: dict[int, list[int]] = {}
    for gi, b in enumerate(bins.tolist()):
        groups.setdefault(int(b), []).append(gi)

    if OUT.exists():
        OUT.unlink()
    with Dataset(OUT, "w", format="NETCDF4") as f:
        f.createDimension("pixels", n)
        f.createDimension("time", N_TIME)
        f.createDimension("bands", 6)
        v = f.createVariable(
            "data", "f4", ("pixels", "time", "bands"),
            chunksizes=(min(1024, n), N_TIME, 6),
            zlib=True, complevel=1, shuffle=True,
        )
        v.comment = (
            "Six optical bands, DN, 3-day nanmedian. "
            "NIR is L30 B05 and S30 B8A. SWIR1 is L30 B06 and S30 B11. "
            "SWIR2 is L30 B07 and S30 B12. Gaps are NaN."
        )
        chunk = 8192
        for p0 in range(0, n, chunk):
            p1 = min(p0 + chunk, n)
            block = np.array(series[p0:p1], dtype=np.float32)
            grid = np.full((p1 - p0, N_TIME, 6), np.nan, np.float32)
            for b, idxs in groups.items():
                take = block[:, idxs, :]
                if len(idxs) == 1:
                    grid[:, b, :] = take[:, 0, :]
                else:
                    with np.errstate(all="ignore"):
                        grid[:, b, :] = np.nanmedian(take, axis=1)
            v[p0:p1] = grid
            log(f"gridded {p1}/{n}")
            del block, grid

        f.createVariable("time", str, ("time",))[:] = np.array(calendar_times(), dtype=object)
        f.createVariable("band", str, ("bands",))[:] = np.array(BANDS6, dtype=object)
        f.createVariable("lon", "f4", ("pixels",))[:] = lon
        f.createVariable("lat", "f4", ("pixels",))[:] = lat
        f.createVariable("row", "i4", ("pixels",))[:] = rows.astype(np.int32)
        f.createVariable("col", "i4", ("pixels",))[:] = cols.astype(np.int32)
        flum = f.createVariable("lum_code", "i2", ("pixels",))
        flum.comment = "61 agriculture, 71 woodland, 72 shrubland, 73 grassland, 74 mangrove. Majority of 10 m cells."
        flum[:] = lum_veg
        f.createVariable("dist_urban_m", "f4", ("pixels",))[:] = dist_veg
        f.setncattr("source_landuse", "Planning Department LUMHK raster 2024, majority-voted to HLS 30 m")
        f.setncattr("crs", "EPSG:32649")
        f.setncattr("urban_codes", ",".join(str(c) for c in URBAN_CODES))
        f.setncattr("note", "Pixel order is row-major on the 1830x1830 window. Positions are row, col, lon, lat.")

    del series
    series_path.unlink(missing_ok=True)
    np.savez_compressed(
        ROOT / "phenology/landuse/hls_window_lum2024_majority.npz",
        lum=lum.astype(np.int16),
    )
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
