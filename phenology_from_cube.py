#!/usr/bin/env python3
"""SOS/EOS on full-grid veg_filled.nc (time,band,y,x). Fast subset via --max-pixels."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from hk_paths import PHENO
from phenology_hplm import N_YEARS, STEPS, YEAR0, evi2, log, metrics_block
from phenology_smooth import SMOOTH_METHODS

FILLED = PHENO / "veg_filled.nc"
OUT = PHENO / "phenology_sos_eos_grid.nc"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smooth", choices=SMOOTH_METHODS, default="sg9")
    p.add_argument("--src", type=Path, default=FILLED)
    p.add_argument("--dst", type=Path, default=OUT)
    p.add_argument("--max-pixels", type=int, default=0, help="0 = all vegetation pixels")
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--batch", type=int, default=512)
    args = p.parse_args()
    if not args.src.exists():
        raise SystemExit(f"missing {args.src}")

    t0 = time.time()
    with Dataset(args.src) as src:
        veg = np.array(src.variables["vegetation"][:], np.uint8) > 0
        ys, xs = np.nonzero(veg)
        if args.max_pixels > 0 and ys.size > args.max_pixels:
            rng = np.random.default_rng(args.seed)
            take = rng.choice(ys.size, args.max_pixels, replace=False)
            ys, xs = ys[take], xs[take]
        n = int(ys.size)
        log(f"pixels {n} smooth={args.smooth}")
        data = src.variables["data"]
        sos = np.full((n, N_YEARS), np.nan, np.float32)
        eos = np.full((n, N_YEARS), np.nan, np.float32)
        from cube_io import TILE

        for b0 in range(0, n, args.batch):
            b1 = min(b0 + args.batch, n)
            yy, xx = ys[b0:b1], xs[b0:b1]
            block = np.empty((b1 - b0, data.shape[0], 6), np.int16)
            buckets: dict[tuple[int, int], list[int]] = {}
            for i, (y, x) in enumerate(zip(yy, xx)):
                buckets.setdefault((int(y) // TILE * TILE, int(x) // TILE * TILE), []).append(i)
            for (y0, x0), idxs in buckets.items():
                y1, x1 = min(y0 + TILE, 1830), min(x0 + TILE, 1830)
                tile = np.array(data[:, :, y0:y1, x0:x1], np.int16)
                rows = yy[idxs] - y0
                cols = xx[idxs] - x0
                block[idxs] = tile[:, :, rows, cols].transpose(2, 0, 1)
            red = block[:, :, 2].astype(np.float32)
            nir = block[:, :, 3].astype(np.float32)
            evi = evi2(red, nir)
            metrics_block(evi, sos[b0:b1], eos[b0:b1], args.smooth)
            log(f"phenology {b1}/{n} ({time.time()-t0:.1f}s)")

        if args.dst.exists():
            args.dst.unlink()
        with Dataset(args.dst, "w", format="NETCDF4") as dst:
            dst.createDimension("pixels", n)
            dst.createDimension("year", N_YEARS)
            dst.createVariable("year", "i2", ("year",))[:] = np.arange(
                YEAR0, YEAR0 + N_YEARS, dtype=np.int16
            )
            dst.createVariable("row", "i4", ("pixels",))[:] = ys
            dst.createVariable("col", "i4", ("pixels",))[:] = xs
            vs = dst.createVariable("sos", "f4", ("pixels", "year"))
            ve = dst.createVariable("eos", "f4", ("pixels", "year"))
            vs[:] = sos
            ve[:] = eos
            dst.setncattr("smooth_method", args.smooth)
            ok = np.isfinite(sos) & np.isfinite(eos)
            dst.setncattr("n_valid_pixel_years", int(ok.sum()))
    log(f"wrote {args.dst} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
