#!/usr/bin/env python3
"""Earlier one-direction trial. Not the locked clean.

The locked pre-fill clean is bolton_clean.apply_bolton_clean: bright anomalies
plus EVI spikes in both directions, neighbor gap under 90 days, threshold 0.1.
This script only removes upward spikes inside 45 days.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from bolton_clean import _gather, _neighbor_index
from hk_paths import PHENO
from phenology_hplm import greenness

TILE = 256
HEIGHT = 1830
SPAN = 45.0
RISE = 0.1
SRC = PHENO / "veg_cube_bolton.nc"
DST = PHENO / "veg_cube_evi45.nc"


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def upward_mask(pixels: np.ndarray, day: np.ndarray) -> np.ndarray:
    valid = (pixels[:, :, 0] > 0) & (pixels[:, :, 2] > 0) & (pixels[:, :, 3] > 0)
    evi = np.where(valid, greenness(pixels, "evi"), np.nan)
    prev, nxt = _neighbor_index(valid)
    has = (prev >= 0) & (nxt < pixels.shape[1]) & valid
    grid = np.broadcast_to(day, valid.shape)
    d0, d1 = _gather(grid, prev, np.nan), _gather(grid, nxt, np.nan)
    v0 = _gather(np.where(valid, evi, 0.0), prev, np.nan)
    v1 = _gather(np.where(valid, evi, 0.0), nxt, np.nan)
    span = d1 - d0
    weight = (day[None, :] - d0) / np.where(span > 0, span, np.nan)
    fitted = v0 * (1.0 - weight) + v1 * weight
    return has & np.isfinite(fitted) & (span > 0) & (span < SPAN) & ((evi - fitted) > RISE) & (evi > v0) & (evi > v1)


def _clean_tile(job: tuple) -> tuple[int, int, np.ndarray, int, int]:
    y0, x0, day, veg_tile = job
    y1, x1 = min(y0 + TILE, HEIGHT), min(x0 + TILE, HEIGHT)
    with Dataset(SRC) as src:
        tile = np.array(src.variables["data"][:, :, y0:y1, x0:x1], np.int16)
    rows, cols = np.nonzero(veg_tile)
    observed = removed = 0
    if rows.size:
        pixels = tile[:, :, rows, cols].transpose(2, 0, 1)
        valid = (pixels[:, :, 0] > 0) & (pixels[:, :, 2] > 0) & (pixels[:, :, 3] > 0)
        drop = upward_mask(pixels, day)
        pixels[drop] = 0
        tile[:, :, rows, cols] = pixels.transpose(1, 2, 0)
        observed = int(valid.sum())
        removed = int(drop.sum())
    return y0, x0, tile, observed, removed


def main() -> None:
    t0 = time.time()
    workers = int(os.environ.get("CLEAN_WORKERS", os.cpu_count() or 8))
    with Dataset(SRC) as src:
        times = pd.to_datetime([str(t) for t in src.variables["time"][:]], format="%Y%j")
        day = times.to_numpy().astype("datetime64[D]").astype(np.float32)
        veg = np.array(src.variables["vegetation"][:], np.uint8) > 0
        if DST.exists():
            DST.unlink()
        with Dataset(DST, "w", format="NETCDF4") as dst:
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
            for attr in src.ncattrs():
                dst.setncattr(attr, src.getncattr(attr))
            dst.setncattr("evi_upward_span_days", SPAN)
            dst.setncattr("evi_upward_rise", RISE)
            dst.setncattr(
                "evi_upward",
                "Dates with EVI more than 0.1 above both clear neighbors are set to 0 when the gap is under 45 days.",
            )
            jobs = []
            for y0 in range(0, HEIGHT, TILE):
                for x0 in range(0, HEIGHT, TILE):
                    y1, x1 = min(y0 + TILE, HEIGHT), min(x0 + TILE, HEIGHT)
                    jobs.append((y0, x0, day, veg[y0:y1, x0:x1]))
            log(f"workers {workers} tiles {len(jobs)}")
            observed = removed = done = 0
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for y0, x0, tile, n_obs, n_rm in pool.map(_clean_tile, jobs, chunksize=1):
                    y1, x1 = min(y0 + TILE, HEIGHT), min(x0 + TILE, HEIGHT)
                    data_out[:, :, y0:y1, x0:x1] = tile
                    observed += n_obs
                    removed += n_rm
                    done += 1
                    log(f"tile {done}/{len(jobs)} removed {removed} observed {observed} ({time.time()-t0:.0f}s)")
            dst.setncattr("evi_upward_removed", int(removed))
            dst.setncattr("evi_upward_observed", int(observed))
    # The known high point should be gone.
    with Dataset(DST) as f:
        dn = np.array(f.variables["data"][:, :, 110, 1799], np.int16)
        times = pd.to_datetime([str(t) for t in f.variables["time"][:]], format="%Y%j")
    hit = np.flatnonzero((times.year == 2016) & (times.dayofyear == 361))
    kept = int(dn[hit, 0].sum()) if hit.size else -1
    log(f"wrote {DST} removed {removed} ({100*removed/max(observed,1):.3f}%) example_2016-12-26_blue_dn {kept}")


if __name__ == "__main__":
    main()
