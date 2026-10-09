#!/usr/bin/env python3
"""Write veg_cube_bolton.nc: locked pre-fill outliers set back to missing.

Bright anomalies follow Bolton et al. (2020). EVI spikes use a 90-day
neighbor gap and a 0.1 threshold in both directions. The source is the
3-day composite. Gap filling is not applied.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from bolton_clean import apply_bolton_clean
from hk_paths import PHENO

TILE = 256
HEIGHT = 1830
SRC = PHENO / "veg_cube.nc"
DST = PHENO / "veg_cube_bolton.nc"
PARTIAL = PHENO / "veg_cube_bolton.nc.partial"


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def _clean_tile(job: tuple) -> tuple[int, int, np.ndarray, dict[str, int]]:
    y0, x0, day, veg_tile = job
    y1, x1 = min(y0 + TILE, HEIGHT), min(x0 + TILE, HEIGHT)
    with Dataset(SRC) as src:
        tile = np.array(src.variables["data"][:, :, y0:y1, x0:x1], np.int16)
    rows, cols = np.nonzero(veg_tile)
    stats = {"observed": 0, "bright": 0, "spike": 0, "removed": 0}
    if rows.size:
        pixels = tile[:, :, rows, cols].transpose(2, 0, 1)
        cleaned, stats = apply_bolton_clean(pixels, day)
        tile[:, :, rows, cols] = cleaned.transpose(1, 2, 0)
    return y0, x0, tile, stats


def main() -> None:
    t0 = time.time()
    workers = int(os.environ.get("CLEAN_WORKERS", os.cpu_count() or 8))
    with Dataset(SRC) as src:
        times = pd.to_datetime([str(t) for t in src.variables["time"][:]], format="%Y%j")
        day = times.to_numpy().astype("datetime64[D]").astype(np.float32)
        veg = np.array(src.variables["vegetation"][:], np.uint8) > 0
        if PARTIAL.exists():
            PARTIAL.unlink()
        with Dataset(PARTIAL, "w", format="NETCDF4") as dst:
            for name, dim in src.dimensions.items():
                dst.createDimension(name, len(dim) if not dim.isunlimited() else None)
            for name, var in src.variables.items():
                if name == "data":
                    continue
                copied = dst.createVariable(name, var.dtype, var.dimensions)
                copied[:] = var[:]
                for attr in var.ncattrs():
                    copied.setncattr(attr, var.getncattr(attr))
            data_out = dst.createVariable(
                "data", "i2", ("time", "band", "y", "x"),
                zlib=True, complevel=1, shuffle=True,
                chunksizes=(4, 6, TILE, TILE), fill_value=0,
            )
            data_out.comment = (
                "HLS DN. Bright anomalies (Bolton et al. 2020) and EVI spikes "
                "beyond 0.1 within a 90-day neighbor gap set to 0. No gap filling."
            )
            totals = {"observed": 0, "bright": 0, "spike": 0, "removed": 0}
            jobs = []
            for y0 in range(0, HEIGHT, TILE):
                for x0 in range(0, HEIGHT, TILE):
                    y1, x1 = min(y0 + TILE, HEIGHT), min(x0 + TILE, HEIGHT)
                    local = veg[y0:y1, x0:x1]
                    if np.any(local):
                        jobs.append((y0, x0, day, local))
            log(f"workers {workers} tiles {len(jobs)}")
            done = 0
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
                for y0, x0, tile, stats in pool.map(_clean_tile, jobs, chunksize=1):
                    y1, x1 = min(y0 + TILE, HEIGHT), min(x0 + TILE, HEIGHT)
                    data_out[:, :, y0:y1, x0:x1] = tile
                    for key in totals:
                        totals[key] += stats[key]
                    done += 1
                    log(f"tile {done}/{len(jobs)} {totals} ({time.time() - t0:.0f}s)")
            for attr in src.ncattrs():
                dst.setncattr(attr, src.getncattr(attr))
            dst.setncattr(
                "bolton_clean",
                "bright anomaly + bidirectional EVI despike, span < 90 days, delta 0.1; no snow fill; no spline",
            )
            dst.setncattr("bolton_reference", "Bolton et al. 2020, Remote Sensing of Environment 240:111685")
            for key, value in totals.items():
                dst.setncattr(f"bolton_{key}", value)
    os.replace(PARTIAL, DST)
    summary = PHENO / "bolton_clean_counts.json"
    summary.write_text(json.dumps(totals, indent=2) + "\n")
    log(f"wrote {DST} and {summary}")


if __name__ == "__main__":
    main()
