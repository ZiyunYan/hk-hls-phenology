#!/usr/bin/env python3
"""Update veg_cube.nc masks: drop agriculture; optional stable veg via 2018∩2024 LUM."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import rasterio
from netCDF4 import Dataset
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

ROOT = Path("/intelnvme03/ziyun218/hls_49QHE_hk")
CUBE = ROOT / "phenology/veg_cube.nc"
META = json.loads((ROOT / "meta.json").read_text())
HEIGHT = WIDTH = 1830
AGRI = 61
VEG_CODES = (71, 72, 73, 74)  # no agriculture
URBAN_CODES = (1, 2, 3, 11, 21, 22, 23, 31, 41, 42, 43, 44)
LUM2018 = ROOT / "phenology/landuse/LUMHK_RasterGrid_2018/LUHK_end2018.tif"


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def hls_transform() -> Affine:
    t = META["transform"]
    return Affine(t[0], t[1], t[2], t[3], t[4], t[5])


def majority_lum(tif: Path) -> np.ndarray:
    aff = hls_transform()
    with rasterio.open(tif) as src:
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


def main() -> None:
    use_stable = LUM2018.is_file()
    with Dataset(CUBE, "r+") as f:
        lum24 = np.array(f.variables["lum_code"][:], np.int16)
        keep = np.isin(lum24, VEG_CODES)
        log(f"drop agriculture {int((lum24 == AGRI).sum())} cells; natural veg {int(keep.sum())}")

        if use_stable:
            lum18 = majority_lum(LUM2018)
            keep18 = np.isin(lum18, VEG_CODES)
            urban_either = np.isin(lum24, URBAN_CODES) | np.isin(lum18, URBAN_CODES)
            keep &= keep18 & ~urban_either
            log(
                f"stable 2018∩2024 natural veg (excl urban either year): {int(keep.sum())}; "
                f"dropped {int((~keep & np.isin(lum24, VEG_CODES)).sum())} vs 2024-only veg"
            )
        else:
            log("LUM 2018 not found; only agriculture removed. See apply_veg_mask note.")

        veg_u8 = keep.astype(np.uint8)
        f.variables["vegetation"][:] = veg_u8
        drop = ~keep
        if not drop.any():
            log("nothing to zero in data cube")
            return
        data = f.variables["data"]
        n_time = data.shape[0]
        for t0 in range(0, n_time, 8):
            t1 = min(t0 + 8, n_time)
            block = np.array(data[t0:t1], copy=True)
            block[:, :, drop] = 0
            data[t0:t1] = block
            log(f"masked time {t1}/{n_time}")

        note = "Agriculture (61) removed. Vegetation mask = 71-74"
        if use_stable:
            note += "; stable pixels: veg in 2018 and 2024, not urban in either year."
        else:
            note += "; LUM 2024 only (no 2018 stable filter)."
        f.setncattr("vegetation_mask_note", note)

    log(f"updated {CUBE}")


if __name__ == "__main__":
    main()
