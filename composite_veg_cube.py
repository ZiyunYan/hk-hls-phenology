#!/usr/bin/env python3
"""Full-grid 3-day optical composite on the HLS window, then vegetation mask.

Per granule: cloud / fill / water masking, |z|>=4 DN filter (HK 6-band scaler),
then merge into 3-day bins with median of valid DN (0 = missing). After all
granules, non-vegetation LUM cells are set to 0. Output int16 cube for mapping
and later gap-fill.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from netCDF4 import Dataset
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

from hk_paths import DATA as ROOT
LUM_TIF = ROOT / "phenology/landuse/LUMHK_RasterGrid_2024/LUM_end2024.tif"
OUT = ROOT / "phenology/veg_cube.nc"
META = json.loads((ROOT / "meta.json").read_text())

HEIGHT = 1830
WIDTH = 1830
YEAR0 = 2015
N_YEARS = 11
STEPS_PER_YEAR = 122
N_TIME = N_YEARS * STEPS_PER_YEAR
BANDS6 = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
SLOT = {"B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6, "B8A": 8, "B11": 11, "B12": 12}
VEG_CODES = (71, 72, 73, 74)  # woodland, shrub, grass, mangrove (no agriculture)
CLOUDISH = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
WATER_BIT = 1 << 5
FILL = -9999
NODATA = 0
Z_MAX = 4.0

MEAN = np.array([405.80286, 627.62134, 511.61603, 2351.3054, 1401.8135, 767.9902], np.float32)
STD = np.array([499.76883, 524.465, 585.40576, 1158.5311, 725.94995, 545.9461], np.float32)


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def hls_transform() -> Affine:
    t = META["transform"]
    return Affine(t[0], t[1], t[2], t[3], t[4], t[5])


def bin_of(stamp: str) -> int:
    year = int(stamp[:4])
    doy = int(stamp[4:7])
    if year < YEAR0 or year >= YEAR0 + N_YEARS:
        return -1
    index = (year - YEAR0) * STEPS_PER_YEAR + (doy - 1) // 3
    if index < 0 or index >= N_TIME:
        return -1
    return index


def calendar_times() -> list[str]:
    times = []
    for year in range(YEAR0, YEAR0 + N_YEARS):
        for step in range(STEPS_PER_YEAR):
            doy = step * 3 + 1
            times.append(f"{year}{doy:03d}")
    return times


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


def optical6_grid(data: np.ndarray, sensor: str) -> np.ndarray:
    if sensor == "L30":
        idx = [SLOT["B02"], SLOT["B03"], SLOT["B04"], SLOT["B05"], SLOT["B06"], SLOT["B07"]]
    elif sensor == "S30":
        idx = [SLOT["B02"], SLOT["B03"], SLOT["B04"], SLOT["B8A"], SLOT["B11"], SLOT["B12"]]
    else:
        raise ValueError(sensor)
    return np.array(data[idx], dtype=np.int16, copy=True)


def zero_incomplete(frame: np.ndarray) -> None:
    bad = (frame == NODATA).any(axis=0)
    frame[:, bad] = NODATA


def process_granule(job: tuple[str, str, int]) -> tuple[int, np.ndarray]:
    granule_id, sensor, bin_index = job
    path = ROOT / "granules" / f"{granule_id}.npz"
    with np.load(path) as z:
        data = z["data"]
        fm = z["fmask"]
    out = optical6_grid(data, sensor)
    bad = (fm == 255) | ((fm & CLOUDISH) != 0) | ((fm & WATER_BIT) != 0)
    out[:, bad] = NODATA
    out[out == FILL] = NODATA
    valid = out != NODATA
    zscore = np.abs((out.astype(np.float32) - MEAN[:, None, None]) / STD[:, None, None])
    out[(zscore >= Z_MAX) & valid] = NODATA
    zero_incomplete(out)
    return bin_index, out


def merge_median(existing: np.ndarray, new: np.ndarray) -> np.ndarray:
    stacked = np.stack([existing, new], axis=0).astype(np.float32)
    stacked[stacked == NODATA] = np.nan
    med = np.nanmedian(stacked, axis=0)
    out = np.zeros((6, HEIGHT, WIDTH), np.int16)
    ok = np.isfinite(med) & (med > 0)
    out[ok] = np.rint(med[ok]).astype(np.int16)
    zero_incomplete(out)
    return out


def granule_jobs() -> list[tuple[str, str, int]]:
    manifest = json.loads((ROOT / "granules.json").read_text())
    jobs = []
    for g in manifest:
        stamp = g["datetime"][:7]
        if not ("2015001" <= stamp <= "2025365"):
            continue
        b = bin_of(stamp)
        if b < 0:
            continue
        jobs.append((g["granule_id"], str(g["sensor"]), b))
    jobs.sort(key=lambda t: (t[2], t[0]))
    return jobs


def main() -> None:
    workers = int(os.environ.get("COMPOSITE_WORKERS", min(32, os.cpu_count() or 8)))
    log(f"workers {workers}, dtype int16, nodata {NODATA}")
    log("majority-vote land use onto 30 m grid")
    lum = majority_landuse()
    veg_mask = np.isin(lum, VEG_CODES)
    log(
        f"vegetation cells {int(veg_mask.sum())}: "
        + ", ".join(f"{c}={int((lum == c).sum())}" for c in VEG_CODES)
    )

    jobs = granule_jobs()
    log(f"granules to composite {len(jobs)}")

    mmap_path = ROOT / "phenology/veg_cube_merge.i16"
    if mmap_path.exists():
        mmap_path.unlink()
    cube = np.memmap(mmap_path, dtype=np.int16, mode="w+", shape=(N_TIME, 6, HEIGHT, WIDTH))
    filled_bins: set[int] = set()

    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(process_granule, j): j for j in jobs}
        for fut in as_completed(futs):
            bin_index, frame = fut.result()
            if bin_index not in filled_bins:
                cube[bin_index] = frame
                filled_bins.add(bin_index)
            else:
                plane = np.array(cube[bin_index], copy=True)
                cube[bin_index] = merge_median(plane, frame)
            done += 1
            if done % 100 == 0 or done == len(jobs):
                cube.flush()
                log(f"merged {done}/{len(jobs)} bins {len(filled_bins)} {(time.perf_counter() - t0) / 60:.1f}min")

    log("applying vegetation mask and writing NetCDF")
    not_veg = ~veg_mask
    if OUT.exists():
        OUT.unlink()
    times = calendar_times()
    with Dataset(OUT, "w", format="NETCDF4") as f:
        f.createDimension("time", N_TIME)
        f.createDimension("band", 6)
        f.createDimension("y", HEIGHT)
        f.createDimension("x", WIDTH)
        v = f.createVariable(
            "data",
            "i2",
            ("time", "band", "y", "x"),
            zlib=True,
            complevel=1,
            shuffle=True,
            chunksizes=(4, 6, 256, 256),
            fill_value=NODATA,
        )
        v.comment = (
            "HLS DN int16; 0 = missing. 3-day median composite. Cloud/snow/shadow/cirrus/water "
            f"masked; |z|>={Z_MAX} removed. Non-vegetation LUM cells are 0."
        )
        for ts in range(0, N_TIME, 16):
            te = min(ts + 16, N_TIME)
            block = np.zeros((te - ts, 6, HEIGHT, WIDTH), np.int16)
            for i, t in enumerate(range(ts, te)):
                if t in filled_bins:
                    block[i] = cube[t]
            block[:, :, not_veg] = NODATA
            v[ts:te] = block
            log(f"wrote time {te}/{N_TIME}")

        f.createVariable("time", str, ("time",))[:] = np.array(times, dtype=object)
        f.createVariable("band", str, ("band",))[:] = np.array(BANDS6, dtype=object)
        flum = f.createVariable("lum_code", "i2", ("y", "x"), zlib=True, complevel=1)
        flum[:] = lum
        flum.comment = "Planning Department LUM 2024 majority on HLS 30 m grid."
        fveg = f.createVariable("vegetation", "u1", ("y", "x"), zlib=True, complevel=1)
        fveg[:] = veg_mask.astype(np.uint8)
        f.setncattr("crs", "EPSG:32649")
        f.setncattr("transform", ",".join(str(x) for x in META["transform"][:6]))
        f.setncattr("nodata", NODATA)
        f.setncattr("zscore_max", Z_MAX)
        f.setncattr("band_mean", ",".join(f"{x:.6f}" for x in MEAN))
        f.setncattr("band_std", ",".join(f"{x:.6f}" for x in STD))
        f.setncattr("composite", "3-day median of valid DN; parallel granule merge")
        f.setncattr("note", "Urban and water excluded. Ready for gap-fill on veg pixels.")

    del cube
    mmap_path.unlink(missing_ok=True)

    lum_path = ROOT / "phenology/landuse/hls_window_lum2024_majority.npz"
    np.savez_compressed(lum_path, lum=lum.astype(np.int16), vegetation=veg_mask.astype(np.uint8))
    log(f"wrote {OUT} and {lum_path} total {(time.perf_counter() - t0) / 60:.1f}min")


if __name__ == "__main__":
    main()
