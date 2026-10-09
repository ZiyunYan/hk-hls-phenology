#!/usr/bin/env python3
"""Score one imputator by hiding clear steps on the cleaned cube.

The same pixels and the same 25% hold-out are used for sl122, sl244 and sl366.
Only hidden steps are scored. Observed steps stay in the input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from hk_paths import IMPUTATOR, PHENO
from impute_fast import YEAR, fill_batch, load_model, lonlat_of
from phenology_hplm import evi

BANDS = ("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")
SRC = PHENO / "veg_cube_bolton.nc"
CKPT_ROOT = Path(__file__).resolve().parent / "imputator_ssl/checkpoints"


def ckpt(seq: int) -> Path:
    return CKPT_ROOT / f"HK-Imputator-optical6-topk5-s1-sl{seq}/checkpoint.pth"


def sample(n: int, seed: int):
    """Read whole tiles. Scattered single-pixel reads decompress a chunk each."""
    rng = np.random.default_rng(seed)
    with Dataset(SRC) as src:
        veg = np.array(src.variables["vegetation"][:], np.uint8) > 0
        times = pd.to_datetime([str(t) for t in src.variables["time"][:]], format="%Y%j")
        tiles = [
            (y0, x0)
            for y0 in range(0, veg.shape[0], 256)
            for x0 in range(0, veg.shape[1], 256)
            if np.any(veg[y0:y0 + 256, x0:x0 + 256])
        ]
        order = rng.permutation(len(tiles))
        chunks = []
        rows_all = []
        cols_all = []
        data = src.variables["data"]
        for k in order:
            y0, x0 = tiles[int(k)]
            y1, x1 = min(y0 + 256, veg.shape[0]), min(x0 + 256, veg.shape[1])
            local = veg[y0:y1, x0:x1]
            rr, cc = np.nonzero(local)
            if rr.size == 0:
                continue
            need = n - sum(c.shape[0] for c in chunks)
            take = rr.size if rr.size <= need else rng.choice(rr.size, size=need, replace=False)
            if rr.size > need:
                rr, cc = rr[take], cc[take]
            block = np.array(data[:, :, y0:y1, x0:x1], np.int16)
            chunks.append(np.ascontiguousarray(block[:, :, rr, cc].transpose(2, 0, 1)))
            rows_all.append(rr + y0)
            cols_all.append(cc + x0)
            if sum(c.shape[0] for c in chunks) >= n:
                break
    raw = np.concatenate(chunks, axis=0)[:n]
    return raw, times, np.concatenate(rows_all)[:n].astype(np.int32), np.concatenate(cols_all)[:n].astype(np.int32)


def holdout_mask(raw: np.ndarray, seed: int) -> np.ndarray:
    observed = np.all(raw > 0, axis=-1)
    hide = np.zeros(observed.shape, np.bool_)
    rng = np.random.default_rng(seed)
    for i in range(raw.shape[0]):
        idx = np.flatnonzero(observed[i])
        if idx.size == 0:
            continue
        k = max(1, int(round(0.25 * idx.size)))
        hide[i, rng.choice(idx, size=min(k, idx.size), replace=False)] = True
    return hide


def score(seq: int, n: int, seed: int, batch: int) -> dict:
    import torch

    raw, times, rows, cols = sample(n, seed)
    hide = holdout_mask(raw, seed)
    hidden = raw.copy()
    hidden[hide] = 0
    with Dataset(IMPUTATOR / "train.nc") as f:
        mean = np.array(f.variables["band_mean"][:], np.float32)
        std = np.array(f.variables["band_std"][:], np.float32)
    std = np.where(std == 0, 1.0, std)
    from impute_fast import time_features

    stamp = time_features(times, freq="rs").T.astype(np.float32)
    lon, lat = lonlat_of(rows, cols)
    device = torch.device("cuda:0")
    import impute_fast

    impute_fast.SEQ = seq
    models = {seq: load_model(device, ckpt(seq), seq)}
    if seq == 244:
        models[122] = load_model(device, ckpt(122), 122)
    parts = []
    for start in range(0, n, batch):
        sl = slice(start, start + batch)
        parts.append(fill_batch(models, hidden[sl], stamp, lon[sl], lat[sl], mean, std, device, seq))
    pred = np.concatenate(parts, axis=0)
    obs = raw.astype(np.float32)
    hat = pred.astype(np.float32)
    band_mae = {}
    for b, name in enumerate(BANDS):
        band_mae[name] = float(np.mean(np.abs(hat[:, :, b] - obs[:, :, b])[hide]))
    evi_obs = evi(obs[:, :, 0], obs[:, :, 2], obs[:, :, 3])
    evi_hat = evi(hat[:, :, 0], hat[:, :, 2], hat[:, :, 3])
    ok = hide & np.isfinite(evi_obs) & np.isfinite(evi_hat)
    return {
        "seq": seq,
        "pixels": n,
        "held_out_steps": int(hide.sum()),
        "band_mae_dn": band_mae,
        "mean_band_mae_dn": float(np.mean(list(band_mae.values()))),
        "evi_mae": float(np.mean(np.abs(evi_hat - evi_obs)[ok])),
        "source": str(SRC),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, choices=(122, 244, 366), required=True)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = score(args.seq, args.n, args.seed, args.batch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
