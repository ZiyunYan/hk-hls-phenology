#!/usr/bin/env python3
"""Quick compare smooth methods on random vegetation pixels (~seconds)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from cube_io import load_veg_pixels
from phenology_hplm import N_YEARS, STEPS, evi2, metrics_one_year
from phenology_smooth import SMOOTH_METHODS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--methods", nargs="*", default=list(SMOOTH_METHODS))
    args = p.parse_args()
    t0 = time.time()
    yy, xx, block = load_veg_pixels(args.n, args.seed)
    evi = evi2(block[:, :, 2].astype(np.float32), block[:, :, 3].astype(np.float32))
    print(f"source pixels={evi.shape[0]} time={evi.shape[1]} ({time.time()-t0:.2f}s load)")
    print(f"{'method':<18} {'valid_SOS':>10} {'valid_EOS':>10} {'both':>10}")
    for method in args.methods:
        sos_n = eos_n = both = 0
        for i in range(evi.shape[0]):
            for year in range(N_YEARS):
                y = evi[i, year * STEPS : (year + 1) * STEPS]
                s, e = metrics_one_year(y, method)
                if np.isfinite(s):
                    sos_n += 1
                if np.isfinite(e):
                    eos_n += 1
                if np.isfinite(s) and np.isfinite(e):
                    both += 1
        total = evi.shape[0] * N_YEARS
        print(f"{method:<18} {sos_n:>10} {eos_n:>10} {both:>10}  ({both/total*100:.1f}% pairs)")
    print(f"done in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
