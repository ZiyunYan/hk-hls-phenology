#!/usr/bin/env python3
"""Fill veg_cube.nc on all visible GPUs.

Same imputator and windows as impute_vegetation.py: 366-step context, 122-step
stride, observed DN kept, gaps replaced by rounded predictions. Vegetation
series are read once by spatial chunk into shared memory, then each GPU fills
a contiguous pixel range at the largest batch that fits.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import Transformer

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "imputator_ssl"))

from hk_paths import CKPT, IMPUTATOR, META_PATH, PHENO  # noqa: E402

SRC = Path(os.environ.get("IMPUTE_SRC", PHENO / "veg_cube_bolton.nc"))
DST = Path(os.environ.get("IMPUTE_DST", PHENO / "veg_filled.nc"))
CKPT_PATH = Path(os.environ.get("IMPUTE_CKPT", CKPT))
TAIL_CKPT = Path(os.environ["IMPUTE_TAIL_CKPT"]) if os.environ.get("IMPUTE_TAIL_CKPT") else None
SCALER_NC = IMPUTATOR / "train.nc"
SEQ = int(os.environ.get("IMPUTE_SEQ", "366"))
YEAR = 122
STRIDE = YEAR
N_YEARS = 11
GPU_OFFSET = int(os.environ.get("IMPUTE_GPU_OFFSET", "0"))
TAG = os.environ.get("IMPUTE_TAG", f"sl{SEQ}")
NODATA = 0
HEIGHT = WIDTH = 1830
TILE = 256


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def cache_dir() -> Path:
    job = os.environ.get("SLURM_JOB_ID", "local")
    path = Path(os.environ.get("IMPUTE_CACHE", f"/dev/shm/hk_impute_{job}_{TAG}"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def configs(seq: int | None = None) -> Namespace:
    seq = SEQ if seq is None else seq
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
        output_attention=False,
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
        seq_len=seq,
        label_len=seq,
    )


def year_window(year_index: int) -> tuple[int, slice]:
    if year_index == 0:
        return 0, slice(0, STRIDE)
    if year_index == N_YEARS - 1:
        return (N_YEARS - 3) * STRIDE, slice(2 * STRIDE, SEQ)
    return (year_index - 1) * STRIDE, slice(STRIDE, 2 * STRIDE)


def tiles() -> list[tuple[int, int, int, int]]:
    out = []
    for y0 in range(0, HEIGHT, TILE):
        for x0 in range(0, WIDTH, TILE):
            out.append((y0, x0, min(y0 + TILE, HEIGHT), min(x0 + TILE, WIDTH)))
    return out


def lonlat_of(ys: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads(META_PATH.read_text())
    t = meta["transform"]
    east = t[2] + t[0] * (xs.astype(np.float64) + 0.5) + t[1] * (ys.astype(np.float64) + 0.5)
    north = t[5] + t[4] * (ys.astype(np.float64) + 0.5) + t[3] * (xs.astype(np.float64) + 0.5)
    transformer = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(east, north)
    return np.asarray(lon, np.float32), np.asarray(lat, np.float32)


def _read_tile(job: tuple) -> int:
    y0, x0, y1, x1, rows, cols, offset, mm_path, shape = job
    series_mm = np.memmap(mm_path, dtype=np.int16, mode="r+", shape=shape)
    with Dataset(SRC) as src:
        block = np.array(src.variables["data"][:, :, y0:y1, x0:x1], dtype=np.int16)
    series = np.ascontiguousarray(block[:, :, rows, cols].transpose(2, 0, 1))
    series_mm[offset : offset + series.shape[0]] = series
    series_mm.flush()
    return int(series.shape[0])


def extract(workers: int) -> None:
    cache = cache_dir()
    done = cache / "extract.done"
    if done.exists():
        log(f"extract already done: {cache}")
        return
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    if not CKPT_PATH.exists():
        raise SystemExit(f"missing {CKPT_PATH}")
    if SEQ == 244 and (TAIL_CKPT is None or not TAIL_CKPT.exists()):
        raise SystemExit(f"sl244 needs the sl122 tail checkpoint, got {TAIL_CKPT}")

    with Dataset(SRC) as src:
        veg = np.array(src.variables["vegetation"][:], np.uint8) > 0
        times = [str(t) for t in src.variables["time"][:]]
    n_time = len(times)
    if n_time != N_YEARS * STRIDE:
        raise SystemExit(f"expected {N_YEARS * STRIDE} steps, got {n_time}")

    jobs = []
    ys_all: list[np.ndarray] = []
    xs_all: list[np.ndarray] = []
    offset = 0
    for y0, x0, y1, x1 in tiles():
        local = np.nonzero(veg[y0:y1, x0:x1])
        if local[0].size == 0:
            continue
        rows = local[0].astype(np.int32)
        cols = local[1].astype(np.int32)
        jobs.append((y0, x0, y1, x1, rows, cols, offset))
        ys_all.append(rows + y0)
        xs_all.append(cols + x0)
        offset += int(rows.size)

    n = offset
    shape = (n, n_time, 6)
    log(f"vegetation pixels {n}; cache {cache}")
    mm_path = cache / "series.int16"
    series = np.memmap(mm_path, dtype=np.int16, mode="w+", shape=shape)
    series.flush()
    del series

    ys = np.concatenate(ys_all).astype(np.int32)
    xs = np.concatenate(xs_all).astype(np.int32)
    np.save(cache / "y.npy", ys)
    np.save(cache / "x.npy", xs)
    lon, lat = lonlat_of(ys, xs)
    np.save(cache / "lon.npy", lon)
    np.save(cache / "lat.npy", lat)
    stamp = time_features(pd.to_datetime(times, format="%Y%j"), freq="rs").T.astype(np.float32)
    np.save(cache / "stamp.npy", stamp)
    with Dataset(SCALER_NC) as f:
        mean = np.array(f.variables["band_mean"][:], np.float32)
        std = np.array(f.variables["band_std"][:], np.float32)
    std = np.where(std == 0, 1.0, std)
    np.savez(cache / "scaler.npz", mean=mean, std=std)
    (cache / "shape.json").write_text(json.dumps({"n": n, "t": n_time, "bands": 6}))

    packed = [(*job, str(mm_path), shape) for job in jobs]
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    log(f"reading {len(packed)} tiles with {workers} workers")
    with ctx.Pool(workers) as pool:
        counts = pool.map(_read_tile, packed, chunksize=1)
    if sum(counts) != n:
        raise SystemExit(f"extract count {sum(counts)} != {n}")
    done.write_text(f"{n}\n")
    log("extract done")


def time_features(dates, freq="rs"):
    from utils.timefeatures import time_features as _tf

    return _tf(dates, freq=freq)


def load_model(device, ckpt: Path | None = None, seq: int | None = None):
    import torch
    from models.Transformer import Model

    ckpt = CKPT_PATH if ckpt is None else Path(ckpt)
    seq = SEQ if seq is None else seq
    model = Model(configs(seq))
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def iter_windows(seq: int):
    """Yield (model_seq, read_start, keep_slice, write_start).

    sl366: 3-year context, write the target year (GitHub IMPUTE_WINDOWS.md).
    sl122: one year in, one year out.
    sl244: five 2-year blocks, then the last year with the sl122 checkpoint.
    """
    if seq == 122:
        for year in range(N_YEARS):
            yield 122, year * YEAR, slice(0, YEAR), year * YEAR
        return
    if seq == 366:
        for year in range(N_YEARS):
            start, part = year_window(year)
            yield 366, start, part, year * YEAR
        return
    if seq == 244:
        for block in range(5):
            yield 244, block * 244, slice(0, 244), block * 244
        yield 122, 10 * YEAR, slice(0, YEAR), 10 * YEAR
        return
    raise ValueError(f"unsupported seq_len {seq}")


def fill_batch(models, raw, stamp, lon, lat, mean, std, device, seq: int | None = None):
    """raw (B, T, 6) int16 DN, 0 missing -> filled int16. Observed values stay.

    models maps a window length to a loaded imputator.
    """
    import torch

    seq = SEQ if seq is None else seq
    x = raw.astype(np.float32)
    x = np.where(x == NODATA, np.nan, x)
    x = (x - mean) / std
    filled = raw.copy()
    ll = np.stack([lon, lat], axis=-1)
    with torch.inference_mode():
        for model_seq, read_start, keep, write_start in iter_windows(seq):
            model = models[model_seq]
            window = x[:, read_start : read_start + model_seq, :]
            take = window.shape[0]
            mark = np.broadcast_to(stamp[read_start : read_start + model_seq], (take, model_seq, stamp.shape[1])).copy()
            llw = np.broadcast_to(ll[:, None, :], (take, model_seq, 2)).copy()
            pred = model(
                torch.from_numpy(np.nan_to_num(window, nan=0.0)).to(device, non_blocking=True),
                time_mark=torch.from_numpy(mark).to(device, non_blocking=True),
                lon_lat=torch.from_numpy(llw).to(device, non_blocking=True),
                mode="pred",
            )
            if isinstance(pred, dict):
                pred = pred["ssl_loss"]
            pred_dn = pred.float().cpu().numpy() * std + mean
            write_len = keep.stop - keep.start
            year = filled[:, write_start : write_start + write_len, :]
            gap = year == NODATA
            piece = np.maximum(np.rint(pred_dn[:, keep, :]), 0).astype(np.int16)
            year = year.copy()
            year[gap] = piece[gap]
            filled[:, write_start : write_start + write_len, :] = year
    return filled


def probe(candidates: list[int]) -> int:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("probe needs CUDA")
    device = torch.device("cuda:0")
    model = load_model(device)
    best = None
    for batch in candidates:
        try:
            x = torch.zeros(batch, SEQ, 6, device=device)
            mark = torch.zeros(batch, SEQ, 2, device=device)
            ll = torch.zeros(batch, SEQ, 2, device=device)
            with torch.inference_mode():
                model(x, time_mark=mark, lon_lat=ll, mode="pred")
            torch.cuda.synchronize()
            best = batch
            log(f"probe batch {batch} fits")
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            log(f"probe batch {batch} OOM")
    if best is None:
        raise SystemExit("no batch size fit in GPU memory")
    print(f"BATCH {best}", flush=True)
    return best


def fill_shard(shard_id: int, n_shards: int, batch_size: int) -> None:
    import torch

    cache = cache_dir()
    flag = cache / f"fill_{shard_id}_of_{n_shards}.done"
    if flag.exists():
        log(f"shard {shard_id} already done")
        return
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    device = torch.device("cuda:0")
    meta = json.loads((cache / "shape.json").read_text())
    n, n_time = int(meta["n"]), int(meta["t"])
    shape = (n, n_time, 6)
    edges = np.linspace(0, n, n_shards + 1, dtype=np.int64)
    start, stop = int(edges[shard_id]), int(edges[shard_id + 1])
    log(f"shard {shard_id}/{n_shards} pixels {start}:{stop} batch {batch_size} on {torch.cuda.get_device_name(0)}")

    series = np.memmap(cache / "series.int16", dtype=np.int16, mode="r+", shape=shape)
    lon = np.load(cache / "lon.npy", mmap_mode="r")
    lat = np.load(cache / "lat.npy", mmap_mode="r")
    stamp = np.load(cache / "stamp.npy")
    scaler = np.load(cache / "scaler.npz")
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)
    model = load_model(device, CKPT_PATH, SEQ)
    models = {SEQ: model}
    if SEQ == 244:
        models[122] = load_model(device, TAIL_CKPT, 122)
        log(f"shard {shard_id} tail checkpoint {TAIL_CKPT}")

    done = 0
    total = stop - start
    batch = batch_size
    while done < total:
        take = min(batch, total - done)
        sl = slice(start + done, start + done + take)
        raw = np.array(series[sl], dtype=np.int16)
        try:
            filled = fill_batch(
                models, raw, stamp, np.array(lon[sl]), np.array(lat[sl]), mean, std, device
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch <= 256:
                raise
            batch //= 2
            torch.cuda.empty_cache()
            log(f"shard {shard_id} OOM, retry batch {batch}")
            continue
        series[sl] = filled
        done += take
        if done == take or done == total or done % (batch_size * 5) < take:
            log(f"shard {shard_id}: {done}/{total}")
    series.flush()
    flag.write_text(f"{start}:{stop} batch={batch}\n")
    log(f"shard {shard_id} done")


def write_cube() -> None:
    cache = cache_dir()
    meta = json.loads((cache / "shape.json").read_text())
    n, n_time = int(meta["n"]), int(meta["t"])
    n_shards = int((cache / "n_shards.txt").read_text()) if (cache / "n_shards.txt").exists() else 0
    if n_shards:
        missing = [i for i in range(n_shards) if not (cache / f"fill_{i}_of_{n_shards}.done").exists()]
        if missing:
            raise SystemExit(f"missing fill shards {missing}")
    shape = (n, n_time, 6)
    ys = np.load(cache / "y.npy")
    xs = np.load(cache / "x.npy")
    index = np.full((HEIGHT, WIDTH), -1, np.int32)
    index[ys, xs] = np.arange(n, dtype=np.int32)
    series = np.memmap(cache / "series.int16", dtype=np.int16, mode="r", shape=shape)

    if DST.exists():
        DST.unlink()
    log(f"writing {DST}")
    with Dataset(SRC) as src, Dataset(DST, "w", format="NETCDF4") as dst:
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
                    chunksizes=(4, 6, TILE, TILE),
                    fill_value=NODATA,
                )
                out.comment = "Gap-filled int16 DN; 0 missing. Observed DN kept."
            else:
                out = dst.createVariable(name, var.dtype, var.dimensions, zlib=True, complevel=1)
                out[:] = var[:]
            if hasattr(var, "comment") and name != "data":
                out.comment = var.comment
        for att in src.ncattrs():
            dst.setncattr(att, src.getncattr(att))
        dst.setncattr("imputator_checkpoint", str(CKPT_PATH))
        dst.setncattr("impute_seq", SEQ)
        dst.setncattr("impute_source", str(SRC))
        notes = {
            122: "1-year window, year by year; gaps only.",
            244: "2-year blocks covering steps 0-1219; last year filled by sl122.",
            366: "3-year window, 1-year output; middle year for interior.",
        }
        dst.setncattr("impute_note", notes[SEQ])
        data_out = dst.variables["data"]
        data_in = src.variables["data"]
        for i, (y0, x0, y1, x1) in enumerate(tiles(), start=1):
            block = np.array(data_in[:, :, y0:y1, x0:x1], dtype=np.int16)
            local = index[y0:y1, x0:x1]
            rows, cols = np.nonzero(local >= 0)
            if rows.size:
                ids = local[rows, cols]
                block[:, :, rows, cols] = series[ids].transpose(1, 2, 0)
            data_out[:, :, y0:y1, x0:x1] = block
            log(f"wrote tile {i}/{len(tiles())}")
    log(f"filled cube {DST} ({DST.stat().st_size / 1e9:.2f} GB)")


def run_all(extract_workers: int, n_shards: int, batch_size: int) -> None:
    py = sys.executable
    script = str(Path(__file__).resolve())
    cache = cache_dir()
    env = os.environ.copy()
    env["IMPUTE_CACHE"] = str(cache)
    subprocess.check_call([py, script, "--stage", "extract", "--extract-workers", str(extract_workers)], env=env)
    if batch_size <= 0:
        probe_env = env.copy()
        probe_env["CUDA_VISIBLE_DEVICES"] = str(GPU_OFFSET)
        out = subprocess.check_output([py, script, "--stage", "probe"], env=probe_env, text=True)
        batch_lines = [ln for ln in out.splitlines() if ln.startswith("BATCH ")]
        if not batch_lines:
            raise SystemExit(f"probe returned no batch size:\n{out}")
        batch_size = int(batch_lines[-1].split()[1])
        log(f"using batch {batch_size}")
    (cache / "n_shards.txt").write_text(str(n_shards))
    (cache / "batch.txt").write_text(str(batch_size))
    procs = []
    for shard in range(n_shards):
        fill_env = env.copy()
        fill_env["CUDA_VISIBLE_DEVICES"] = str(GPU_OFFSET + shard)
        procs.append(
            subprocess.Popen(
                [
                    py, script, "--stage", "fill",
                    "--shard-id", str(shard),
                    "--n-shards", str(n_shards),
                    "--batch-size", str(batch_size),
                ],
                env=fill_env,
            )
        )
    failed = [p.wait() for p in procs]
    if any(code != 0 for code in failed):
        raise SystemExit(f"fill failed: {failed}")
    subprocess.check_call([py, script, "--stage", "write"], env=env)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast Hong Kong vegetation imputation")
    p.add_argument("--stage", choices=("all", "extract", "probe", "fill", "write"), default="all")
    p.add_argument("--extract-workers", type=int, default=32)
    p.add_argument("--n-shards", type=int, default=8)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0, help="0 probes the largest batch that fits")
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--seq", type=int, choices=(122, 244, 366), default=SEQ)
    p.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    p.add_argument("--tail-ckpt", type=Path, default=TAIL_CKPT)
    p.add_argument("--gpu-offset", type=int, default=GPU_OFFSET)
    p.add_argument("--tag", default=TAG)
    return p.parse_args()


def publish(args: argparse.Namespace) -> None:
    """Apply CLI choices to globals and the environment inherited by workers."""
    global SRC, DST, SEQ, CKPT_PATH, TAIL_CKPT, GPU_OFFSET, TAG
    SRC = Path(args.src)
    DST = Path(args.dst)
    SEQ = int(args.seq)
    CKPT_PATH = Path(args.ckpt)
    TAIL_CKPT = Path(args.tail_ckpt) if args.tail_ckpt else None
    GPU_OFFSET = int(args.gpu_offset)
    TAG = args.tag or f"sl{SEQ}"
    os.environ["IMPUTE_SRC"] = str(SRC)
    os.environ["IMPUTE_DST"] = str(DST)
    os.environ["IMPUTE_SEQ"] = str(SEQ)
    os.environ["IMPUTE_CKPT"] = str(CKPT_PATH)
    os.environ["IMPUTE_TAG"] = TAG
    os.environ["IMPUTE_GPU_OFFSET"] = str(GPU_OFFSET)
    if TAIL_CKPT:
        os.environ["IMPUTE_TAIL_CKPT"] = str(TAIL_CKPT)
    else:
        os.environ.pop("IMPUTE_TAIL_CKPT", None)


def main() -> None:
    args = parse_args()
    publish(args)
    log(f"seq {SEQ} src {SRC.name} dst {DST.name} ckpt {CKPT_PATH.parent.name} gpus {GPU_OFFSET}+{args.n_shards}")
    if args.stage == "extract":
        extract(args.extract_workers)
    elif args.stage == "probe":
        probe([16384, 12288, 8192, 6144, 4096, 3072, 2048, 1024])
    elif args.stage == "fill":
        fill_shard(args.shard_id, args.n_shards, args.batch_size)
    elif args.stage == "write":
        write_cube()
    else:
        run_all(args.extract_workers, args.n_shards, args.batch_size)


if __name__ == "__main__":
    main()
