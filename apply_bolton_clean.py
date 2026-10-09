#!/usr/bin/env python3
"""Write veg_cube_bolton.nc: Bolton et al. (2020) outliers set back to missing.

The source is the 3-day composite. Gap filling is not applied.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from bolton_clean import apply_bolton_clean
from hk_paths import PHENO

TILE = 256
HEIGHT = 1830
SRC = PHENO / "veg_cube.nc"
DST = PHENO / "veg_cube_bolton.nc"


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def main() -> None:
    t0 = time.time()
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
            data_in = src.variables["data"]
            data_out = dst.createVariable(
                "data", "i2", ("time", "band", "y", "x"),
                zlib=True, complevel=1, shuffle=True,
                chunksizes=(4, 6, TILE, TILE), fill_value=0,
            )
            data_out.comment = (
                "HLS DN. Bolton et al. (2020) RSE bright-anomaly and EVI2 "
                "despike observations set to 0. No gap filling."
            )
            totals = {"observed": 0, "bright": 0, "spike": 0, "removed": 0}
            tiles = [
                (y0, x0)
                for y0 in range(0, HEIGHT, TILE)
                for x0 in range(0, HEIGHT, TILE)
                if np.any(veg[y0:y0 + TILE, x0:x0 + TILE])
            ]
            for done, (y0, x0) in enumerate(tiles, start=1):
                y1, x1 = min(y0 + TILE, HEIGHT), min(x0 + TILE, HEIGHT)
                tile = np.array(data_in[:, :, y0:y1, x0:x1], np.int16)
                local = veg[y0:y1, x0:x1]
                rows, cols = np.nonzero(local)
                pixels = tile[:, :, rows, cols].transpose(2, 0, 1)
                cleaned, stats = apply_bolton_clean(pixels, day)
                tile[:, :, rows, cols] = cleaned.transpose(1, 2, 0)
                data_out[:, :, y0:y1, x0:x1] = tile
                for key in totals:
                    totals[key] += stats[key]
                log(f"tile {done}/{len(tiles)} {totals} ({time.time() - t0:.0f}s)")
            for attr in src.ncattrs():
                dst.setncattr(attr, src.getncattr(attr))
            dst.setncattr("bolton_clean", "bright anomaly + EVI2 despike; no snow fill; no spline")
            dst.setncattr("bolton_reference", "Bolton et al. 2020, Remote Sensing of Environment 240:111685")
            for key, value in totals.items():
                dst.setncattr(f"bolton_{key}", value)
    summary = PHENO / "bolton_clean_counts.json"
    summary.write_text(json.dumps(totals, indent=2) + "\n")
    log(f"wrote {DST} and {summary}")


if __name__ == "__main__":
    main()
