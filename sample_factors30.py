#!/usr/bin/env python3
"""Sample 30 m Hong Kong terrain and climate rasters onto the vegetation pixels.

The climate layers are Morgan and Guénard (2019): Lands Department 5 m terrain
upscaled to 30 m, and HKO station climate regressed onto that terrain.
They are long-term monthly means, not a separate map for each year.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

ROOT = Path("/intelnvme03/ziyun218/hls_49QHE_hk")
FACTORS = ROOT / "phenology/factors30"
LUM = ROOT / "phenology/landuse/LUMHK_RasterGrid_2024/LUM_end2024.tif"
META = json.loads((ROOT / "meta.json").read_text())
VEG = (61, 71, 72, 73, 74)
LAYERS = (
    "elevation.tif",
    "slope.tif",
    "waterdist.tif",
    "urbanicity_gauss10.tif",
    "avars_tmean_mean.tif",
    "avars_windsp_mean.tif",
    "biovars_t_11.tif",
    "biovars_t_12.tif",
    "biovars_t_13.tif",
)


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def hls_grid() -> tuple[Affine, np.ndarray, np.ndarray]:
    t = META["transform"]
    aff = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    with rasterio.open(LUM) as src:
        lum = np.full((1830, 1830), 0, np.int16)
        reproject(
            source=src.read(1),
            destination=lum,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=-128,
            dst_transform=aff,
            dst_crs="EPSG:32649",
            dst_nodata=0,
            resampling=Resampling.mode,
        )
    rows, cols = np.nonzero(np.isin(lum, VEG))
    return aff, rows.astype(np.int32), cols.astype(np.int32)


def sample_layer(path: Path, aff: Affine, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as src:
        raw = src.read(1)
        arr = np.full(raw.shape, np.nan, np.float32)
        if np.issubdtype(raw.dtype, np.floating):
            valid = np.isfinite(raw)
            if src.nodata is not None and np.isfinite(src.nodata):
                valid &= raw != src.nodata
            arr[valid] = raw[valid]
        else:
            valid = np.ones(raw.shape, dtype=bool)
            if src.nodata is not None:
                valid &= raw != src.nodata
            arr[valid] = raw[valid]
        dst = np.full((1830, 1830), np.nan, np.float32)
        reproject(
            source=arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=aff,
            dst_crs="EPSG:32649",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return dst[rows, cols]


def main() -> None:
    aff, rows, cols = hls_grid()
    log(f"vegetation pixels {rows.size}")
    out = {"row": rows, "col": cols}
    for name in LAYERS:
        path = FACTORS / name
        if not path.exists():
            log(f"missing {name}")
            continue
        values = sample_layer(path, aff, rows, cols)
        key = path.stem
        out[key] = values.astype(np.float32)
        finite = np.isfinite(values).mean()
        log(f"{key} finite {finite:.3f} median {np.nanmedian(values):.3f}")
    monthly = FACTORS / "monthly"
    if monthly.is_dir():
        for path in sorted(monthly.glob("*.tif")):
            stem = path.stem.lower()
            if stem.startswith("._"):
                continue
            if not (stem.startswith("tmean_") or stem.startswith("prec_") or stem.startswith("windsp_")):
                continue
            values = sample_layer(path, aff, rows, cols)
            out[path.stem] = values.astype(np.float32)
            log(f"monthly {path.stem}")
    dest = FACTORS / "veg_factors.npz"
    np.savez_compressed(dest, **out)
    log(f"wrote {dest} keys {len(out)}")


if __name__ == "__main__":
    main()
