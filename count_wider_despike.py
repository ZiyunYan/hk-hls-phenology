#!/usr/bin/env python3
"""Count a second EVI2 despike on the already cleaned spectral cube.

Same test as Bolton et al. (2020): more than 0.1 below the linear bridge and
below both neighboring clear observations. The 45-day cap is lifted so the
counts can be split by gap.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from bolton_clean import BLUE, RED, NIR, _gather, _neighbor_index, evi2_from_dn
from hk_paths import PHENO

TILE = 256
HEIGHT = 1830
SRC = PHENO / "veg_cube_bolton.nc"
EDGES = (45, 60, 90, 120, 180)


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def main() -> None:
    t0 = time.time()
    with Dataset(SRC) as src:
        times = pd.to_datetime([str(t) for t in src.variables["time"][:]], format="%Y%j")
        day = times.to_numpy().astype("datetime64[D]").astype(np.float32)
        veg = np.array(src.variables["vegetation"][:], np.uint8) > 0
        data = src.variables["data"]
        observed = 0
        bins = {edge: 0 for edge in EDGES}
        bins[10_000] = 0
        tiles = [
            (y0, x0)
            for y0 in range(0, HEIGHT, TILE)
            for x0 in range(0, HEIGHT, TILE)
            if np.any(veg[y0:y0 + TILE, x0:x0 + TILE])
        ]
        for done, (y0, x0) in enumerate(tiles, start=1):
            y1, x1 = min(y0 + TILE, HEIGHT), min(x0 + TILE, HEIGHT)
            tile = np.array(data[:, :, y0:y1, x0:x1], np.int16)
            rows, cols = np.nonzero(veg[y0:y1, x0:x1])
            pixels = tile[:, :, rows, cols].transpose(2, 0, 1)
            del tile
            valid = (pixels[:, :, BLUE] > 0) & (pixels[:, :, RED] > 0) & (pixels[:, :, NIR] > 0)
            observed += int(valid.sum())
            evi = np.where(valid, evi2_from_dn(pixels[:, :, RED], pixels[:, :, NIR]), np.nan)
            prev, nxt = _neighbor_index(valid)
            has = (prev >= 0) & (nxt < pixels.shape[1]) & valid
            day_grid = np.broadcast_to(day, valid.shape)
            day_prev = _gather(day_grid, prev, np.nan)
            day_next = _gather(day_grid, nxt, np.nan)
            evi_prev = _gather(np.where(valid, evi, 0.0), prev, np.nan)
            evi_next = _gather(np.where(valid, evi, 0.0), nxt, np.nan)
            span = day_next - day_prev
            weight = (day[None, :] - day_prev) / np.where(span > 0, span, np.nan)
            fitted = evi_prev * (1.0 - weight) + evi_next * weight
            cand = has & np.isfinite(fitted) & ((fitted - evi) > 0.1) & (evi_prev > evi) & (evi_next > evi)
            values = span[cand]
            lower = 0.0
            for edge in list(EDGES) + [10_000]:
                bins[edge] += int(np.sum((values >= lower) & (values < edge)))
                lower = edge
            removed_120 = sum(bins[edge] for edge in EDGES if edge <= 120)
            log(
                f"tile {done}/{len(tiles)} observed {observed} "
                f"extra<120 {removed_120} ({time.time() - t0:.0f}s)"
            )
    labels = ["<45", "45–60", "60–90", "90–120", "120–180", "≥180"]
    keys = [45, 60, 90, 120, 180, 10_000]
    print("observed", observed)
    running = 0
    for label, key in zip(labels, keys):
        running += bins[key]
        print(f"{label:8} {bins[key]:10d}  cumulative {running}")
    extra_120 = sum(bins[k] for k in (45, 60, 90, 120))
    print(f"second pass span<120 {extra_120} ({100 * extra_120 / max(observed, 1):.3f}% of remaining)")


if __name__ == "__main__":
    main()
