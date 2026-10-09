#!/usr/bin/env python3
"""Gap-fill veg_cube.nc on the 30 m grid with the Hong Kong imputator (366-day windows)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from netCDF4 import Dataset
from rasterio.transform import Affine
from rasterio.warp import transform as rio_transform

SSL = Path(__file__).resolve().parent / "imputator_ssl"
sys.path.insert(0, str(SSL))
from models.Transformer import Model  # noqa: E402
from utils.timefeatures import time_features  # noqa: E402

from hk_paths import CKPT, IMPUTATOR, META_PATH, PHENO  # noqa: E402

ROOT = PHENO
SRC = ROOT / "veg_cube.nc"
DST = ROOT / "veg_filled.nc"
SHARD_DIR = ROOT / "impute_shards"
SCALER_NC = IMPUTATOR / "train.nc"
META = json.loads(META_PATH.read_text())
SEQ = 366
STRIDE = 122
N_YEARS = 11
NODATA = 0
HEIGHT = WIDTH = 1830


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def configs() -> Namespace:
    return Namespace(
        enc_in=6,
        d_model=256,
        n_heads=8,
        e_layers=6,
        d_ff=1024,
        embed="timeF",
        freq="rs",
        dropout=0.0,
        factor=1,
        output_attention=True,
        activation="gelu",
        mask_rate=0.8,
        imp_n_storage_tokens=2,
        lon_lat_n_fourier_freqs=4,
        use_lon_lat_embed=1,
        geo_dropout_p=0.5,
        imp_rec_loss="mse",
        imp_huber_delta=1.0,
        imp_rec_alpha=1.0,
        imp_smooth_beta=1.0,
        imp_smooth_mode="dy2",
        imp_trim_topk_per_seq=5,
        imp_trim_min_keep=8,
        imp_mask_min_p=0.4,
        seq_len=SEQ,
        label_len=SEQ,
    )


def year_window(year_index: int) -> tuple[int, slice]:
    if year_index == 0:
        return 0, slice(0, STRIDE)
    if year_index == N_YEARS - 1:
        return (N_YEARS - 3) * STRIDE, slice(2 * STRIDE, SEQ)
    return (year_index - 1) * STRIDE, slice(STRIDE, 2 * STRIDE)


def lonlat_grid() -> tuple[np.ndarray, np.ndarray]:
    t = META["transform"]
    aff = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    ys, xs = np.meshgrid(np.arange(HEIGHT), np.arange(WIDTH), indexing="ij")
    xmap, ymap = rasterio.transform.xy(aff, ys.ravel(), xs.ravel(), offset="center")
    lon, lat = rio_transform("EPSG:32649", "EPSG:4326", np.asarray(xmap), np.asarray(ymap))
    return np.asarray(lon, np.float32).reshape(HEIGHT, WIDTH), np.asarray(lat, np.float32).reshape(HEIGHT, WIDTH)


def load_model(device: torch.device) -> nn.Module:
    model = Model(configs())
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    model.eval()
    return model.to(device)


def fill_batch(
    model: nn.Module,
    raw: np.ndarray,
    stamp: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """raw (B, T, 6) int16 DN, 0 missing -> filled int16."""
    x = raw.astype(np.float32)
    miss = x == NODATA
    x = np.where(miss, np.nan, x)
    x = (x - mean) / std
    filled = raw.copy()
    ll = np.stack([lon, lat], axis=-1)
    with torch.inference_mode():
        for year_index in range(N_YEARS):
            start, part = year_window(year_index)
            window = x[:, start : start + SEQ, :]
            take = window.shape[0]
            mark = np.broadcast_to(stamp[start : start + SEQ], (take, SEQ, stamp.shape[1])).copy()
            llw = np.broadcast_to(ll[:, None, :], (take, SEQ, 2)).copy()
            pred = model(
                torch.from_numpy(np.nan_to_num(window, nan=0.0)).to(device, non_blocking=True),
                time_mark=torch.from_numpy(mark).to(device, non_blocking=True),
                lon_lat=torch.from_numpy(llw).to(device, non_blocking=True),
                mode="pred",
            )
            if isinstance(pred, dict):
                pred = pred["ssl_loss"]
            pred_dn = pred.float().cpu().numpy() * std + mean
            year = filled[:, year_index * STRIDE : (year_index + 1) * STRIDE, :]
            gap = year == NODATA
            piece = np.rint(pred_dn[:, part, :]).astype(np.int16)
            piece = np.maximum(piece, 0)
            year = year.copy()
            year[gap] = piece[gap]
            filled[:, year_index * STRIDE : (year_index + 1) * STRIDE, :] = year
    return filled


def prepare_output() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    if DST.exists():
        log(f"output exists, skip prepare: {DST}")
        return
    with Dataset(SRC) as src:
        times = [str(t) for t in src.variables["time"][:]]
        n_time = len(times)
        with Dataset(DST, "w", format="NETCDF4") as dst:
            for name, dim in src.dimensions.items():
                dst.createDimension(name, len(dim) if not dim.isunlimited() else None)
            for name, var in src.variables.items():
                if name == "data":
                    out = dst.createVariable(
                        name,
                        "i2",
                        var.dimensions,
                        zlib=True,
                        complevel=1,
                        shuffle=True,
                        chunksizes=(4, 6, 256, 256),
                        fill_value=NODATA,
                    )
                    out.comment = "Gap-filled int16 DN; 0 missing. Observed DN kept."
                else:
                    out = dst.createVariable(name, var.dtype, var.dimensions, zlib=True, complevel=1)
                if hasattr(var, "comment"):
                    out.comment = var.comment
            for att in src.ncattrs():
                dst.setncattr(att, src.getncattr(att))
            dst.setncattr("imputator_checkpoint", str(CKPT))
            dst.setncattr("impute_note", "3-year window, 1-year stride; middle year for interior.")
            data_in = src.variables["data"]
            data_out = dst.variables["data"]
            log("copying source cube to output")
            for t0 in range(0, n_time, 32):
                t1 = min(t0 + 32, n_time)
                data_out[t0:t1] = data_in[t0:t1]
                log(f"copied time {t1}/{n_time}")
    log(f"prepared {DST}")


def shard_path(shard_id: int, n_shards: int) -> Path:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    return SHARD_DIR / f"shard_{shard_id}_of_{n_shards}.npz"


def run_fill(shard_id: int, n_shards: int, gpu: int, batch_size: int) -> None:
    if not CKPT.exists():
        raise SystemExit(f"checkpoint not ready: {CKPT}")
    if not DST.exists():
        raise SystemExit(f"run --stage prepare first (missing {DST})")
    out_npz = shard_path(shard_id, n_shards)
    if out_npz.exists():
        log(f"shard {shard_id} already done: {out_npz}")
        return

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    device = torch.device(f"cuda:{gpu}")
    log(f"shard {shard_id}/{n_shards} gpu {gpu} batch {batch_size}")

    with Dataset(SCALER_NC) as f:
        mean = np.array(f.variables["band_mean"][:], np.float32)
        std = np.array(f.variables["band_std"][:], np.float32)
    std = np.where(std == 0, 1.0, std)

    model = load_model(device)
    lon2d, lat2d = lonlat_grid()

    with Dataset(SRC) as src:
        veg = np.array(src.variables["vegetation"][:], np.uint8) > 0
        ys, xs = np.nonzero(veg)
        order = np.arange(ys.size)
        order = order[shard_id::n_shards]
        ys, xs = ys[order], xs[order]
        n = int(ys.size)
        log(f"vegetation pixels in shard {n}")
        times = [str(t) for t in src.variables["time"][:]]
        stamp = time_features(pd.to_datetime(times, format="%Y%j"), freq="rs").T.astype(np.float32)
        data_in = src.variables["data"]
        n_time = len(times)

        yy_all: list[np.ndarray] = []
        xx_all: list[np.ndarray] = []
        chunks: list[np.ndarray] = []
        batch = batch_size
        done = 0
        while done < n:
            take = min(batch, n - done)
            yy = ys[done : done + take]
            xx = xs[done : done + take]
            raw = np.array(data_in[:, :, yy, xx], dtype=np.int16).transpose(2, 0, 1)
            lon = lon2d[yy, xx]
            lat = lat2d[yy, xx]
            try:
                filled = fill_batch(model, raw, stamp, lon, lat, mean, std, device)
            except torch.cuda.OutOfMemoryError:
                if batch <= 256:
                    raise
                batch //= 2
                torch.cuda.empty_cache()
                log(f"OOM, retry batch {batch}")
                continue
            chunks.append(filled.transpose(1, 2, 0))  # (T, 6, take)
            yy_all.append(yy)
            xx_all.append(xx)
            done += take
            log(f"shard {shard_id}: filled {done}/{n}")

        data = np.concatenate(chunks, axis=2)
        y = np.concatenate(yy_all)
        x = np.concatenate(xx_all)
        np.savez_compressed(out_npz, y=y, x=x, data=data, n_time=n_time)
        log(f"wrote {out_npz} ({data.nbytes / 1e9:.2f} GB raw axis)")


def run_merge(n_shards: int) -> None:
    if not DST.exists():
        raise SystemExit(f"missing {DST}")
    for i in range(n_shards):
        if not shard_path(i, n_shards).exists():
            raise SystemExit(f"missing shard {i}")
    with Dataset(DST, "r+") as dst:
        data_out = dst.variables["data"]
        for i in range(n_shards):
            z = np.load(shard_path(i, n_shards))
            yy, xx, block = z["y"], z["x"], z["data"]
            log(f"merge shard {i}: {yy.size} pixels")
            data_out[:, :, yy, xx] = block
    log(f"merged into {DST}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Impute veg_cube.nc -> veg_filled.nc")
    p.add_argument(
        "--stage",
        choices=("prepare", "fill", "merge", "all"),
        default="all",
        help="prepare=copy nc; fill=GPU shard; merge=apply shards; all=single-GPU legacy path",
    )
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8192)
    return p.parse_args()


def run_all_single_gpu(gpu: int, batch_size: int) -> None:
    """One process, one GPU (no DataParallel)."""
    prepare_output()
    run_fill(0, 1, gpu, batch_size)
    if shard_path(0, 1).exists():
        run_merge(1)
        shard_path(0, 1).unlink()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare_output()
    elif args.stage == "fill":
        run_fill(args.shard_id, args.n_shards, args.gpu, args.batch_size)
    elif args.stage == "merge":
        run_merge(args.n_shards)
    elif args.stage == "all":
        if args.n_shards > 1:
            raise SystemExit("use run_impute_fast.sh for multi-GPU (prepare + fill shards + merge)")
        run_all_single_gpu(args.gpu, args.batch_size)
    else:
        raise SystemExit(f"unknown stage {args.stage}")


if __name__ == "__main__":
    main()
