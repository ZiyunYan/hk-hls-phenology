"""Fast tiled reads from (time,band,y,x) phenology cubes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from hk_paths import PHENO

TILE = 256
RED, NIR = 2, 3


def load_veg_pixels(
    n: int,
    seed: int,
    src: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return yy, xx, data (N, T, 6) int16 from veg_cube or veg_filled."""
    filled = PHENO / "veg_filled.nc"
    cube = PHENO / "veg_cube.nc"
    path = src or (filled if filled.exists() else cube)
    if not path.exists():
        raise FileNotFoundError(path)
    with Dataset(path) as f:
        veg = np.array(f.variables["vegetation"][:], np.uint8) > 0
        ys, xs = np.nonzero(veg)
        take = np.random.default_rng(seed).choice(ys.size, min(n, ys.size), replace=False)
        yy = ys[take].astype(np.int32)
        xx = xs[take].astype(np.int32)
        n_time = f.dimensions["time"].size
        out = np.empty((yy.size, n_time, 6), np.int16)
        buckets: dict[tuple[int, int], list[int]] = {}
        for i, (y, x) in enumerate(zip(yy, xx)):
            key = (int(y) // TILE * TILE, int(x) // TILE * TILE)
            buckets.setdefault(key, []).append(i)
        data = f.variables["data"]
        for (y0, x0), idxs in buckets.items():
            y1, x1 = min(y0 + TILE, 1830), min(x0 + TILE, 1830)
            tile = np.array(data[:, :, y0:y1, x0:x1], np.int16)
            rows = yy[idxs] - y0
            cols = xx[idxs] - x0
            out[idxs] = tile[:, :, rows, cols].transpose(2, 0, 1)
    return yy, xx, out
