#!/usr/bin/env python3
"""Compare the USA 7-band Imputator with the Hong Kong 6-band Imputator.

The USA checkpoint was trained on Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI
with the Dataset_HLS z-score in TimeSeries_SSL_USA/data_provider/data_loader.py.
NDVI is (NIR-Red)/(NIR+Red) on DN, which matches reflectance NDVI. Missing
steps stay NaN so pred mode treats them as unobserved.

Hold-out hides 25% of fully observed steps and scores the prediction there.
The natural-gap pass keeps every real observation and only fills DN==0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import Transformer

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from hk_paths import IMPUTATOR, META_PATH, PHENO  # noqa: E402

USA_ROOT = Path("/work/projects/resilientia/ziyun/TimeSeries_SSL_USA")
HK_SSL = REPO / "imputator_ssl"
USA_CKPT = USA_ROOT / "checkpoints/models/imputator.pth"
HK_CKPT = HK_SSL / "checkpoints/HK-Imputator-optical6-topk5-s1-sl366/checkpoint.pth"
SRC = PHENO / "veg_cube.nc"
OUT_DIR = PHENO / "imputator_compare"
FIG_DIR = REPO / "figures"

SEQ = 366
STRIDE = 122
N_YEARS = 11
HEIGHT = WIDTH = 1830
TILE = 256
HOLDOUT = 0.25
BANDS6 = ("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")
# Dataset_HLS.pre_scaler: the z-score the USA Imputator consumed in training.
USA_MEAN = np.array(
    [4.0530856e02, 6.7968939e02, 7.3541718e02, 2.5394734e03, 2.0182101e03, 1.2844141e03, 5.2847379e-01],
    np.float32,
)
USA_STD = np.array(
    [2.7406531e02, 3.4935846e02, 5.2149530e02, 9.8295978e02, 9.3158044e02, 8.0511346e02, 2.8968227e-01],
    np.float32,
)


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def year_window(year_index: int) -> tuple[int, slice]:
    if year_index == 0:
        return 0, slice(0, STRIDE)
    if year_index == N_YEARS - 1:
        return (N_YEARS - 3) * STRIDE, slice(2 * STRIDE, SEQ)
    return (year_index - 1) * STRIDE, slice(STRIDE, 2 * STRIDE)


def day_features(times: pd.DatetimeIndex) -> np.ndarray:
    day = np.asarray(times.dayofyear) - 1
    span = np.asarray(times.is_leap_year).astype(np.float32) + 365.0
    angle = 2.0 * np.pi * day / span
    return np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)


def lonlat_of(ys: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads(META_PATH.read_text())
    t = meta["transform"]
    east = t[2] + t[0] * (xs.astype(np.float64) + 0.5) + t[1] * (ys.astype(np.float64) + 0.5)
    north = t[5] + t[4] * (ys.astype(np.float64) + 0.5) + t[3] * (xs.astype(np.float64) + 0.5)
    transformer = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(east, north)
    return np.asarray(lon, np.float32), np.asarray(lat, np.float32)


def hk_scaler() -> tuple[np.ndarray, np.ndarray]:
    with Dataset(IMPUTATOR / "train.nc") as f:
        mean = np.array(f.variables["band_mean"][:], np.float32)
        std = np.array(f.variables["band_std"][:], np.float32)
    return mean, np.where(std == 0, 1.0, std)


def model_configs(enc_in: int) -> Namespace:
    return Namespace(
        enc_in=enc_in,
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
        imp_smooth_beta=0.5,
        imp_smooth_mode="dy2",
        imp_trim_topk_per_seq=5,
        imp_trim_min_keep=8,
        imp_mask_min_p=0.4,
        seq_len=SEQ,
        label_len=SEQ,
    )


def load_model(which: str, device):
    import torch
    from models.Transformer import Model

    ckpt = USA_CKPT if which == "usa" else HK_CKPT
    enc_in = 7 if which == "usa" else 6
    model = Model(model_configs(enc_in))
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def with_ndvi(raw: np.ndarray) -> np.ndarray:
    """raw (B, T, 6) DN, 0 missing -> float (B, T, 7) with NaN gaps and NDVI."""
    x = raw.astype(np.float32)
    out = np.concatenate([x, np.full(x.shape[:2] + (1,), np.nan, np.float32)], axis=-1)
    out[:, :, :6] = np.where(raw > 0, out[:, :, :6], np.nan)
    red = out[:, :, 2]
    nir = out[:, :, 3]
    ok = np.isfinite(red) & np.isfinite(nir) & ((red + nir) > 0)
    out[:, :, 6] = np.where(ok, (nir - red) / (nir + red), np.nan)
    return out


def scale(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)


def predict(model, scaled: np.ndarray, stamp: np.ndarray, lon: np.ndarray, lat: np.ndarray, device) -> np.ndarray:
    """scaled (B, T, C) with NaN missing -> prediction in the same scaled space."""
    import torch

    n, t, channels = scaled.shape
    pred = np.empty((n, t, channels), np.float32)
    ll = np.stack([lon, lat], axis=-1).astype(np.float32)
    with torch.inference_mode():
        for year_index in range(N_YEARS):
            start, part = year_window(year_index)
            window = scaled[:, start : start + SEQ, :]
            take = window.shape[0]
            mark = np.broadcast_to(stamp[start : start + SEQ], (take, SEQ, 2)).copy()
            llw = np.broadcast_to(ll[:, None, :], (take, SEQ, 2)).copy()
            out = model(
                torch.from_numpy(window).to(device, non_blocking=True),
                time_mark=torch.from_numpy(mark).to(device, non_blocking=True),
                lon_lat=torch.from_numpy(llw).to(device, non_blocking=True),
                mode="pred",
            )
            pred[:, year_index * STRIDE : (year_index + 1) * STRIDE, :] = out.float().cpu().numpy()[:, part, :]
    return pred


def hide(raw: np.ndarray, holdout: np.ndarray) -> np.ndarray:
    hidden = raw.copy()
    hidden[holdout] = 0
    return hidden


def prepare(n_tiles: int, per_tile: int, n_plot: int, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with Dataset(SRC) as src:
        veg = np.array(src.variables["vegetation"][:], np.uint8) > 0
        times = pd.to_datetime([str(t) for t in src.variables["time"][:]], format="%Y%j")
    if len(times) != N_YEARS * STRIDE:
        raise SystemExit(f"expected {N_YEARS * STRIDE} steps, got {len(times)}")

    plot_ys, plot_xs = np.nonzero(veg)
    # Same draw as plot_index_sample.py: one Generator, one choice, this seed.
    plot_take = np.random.default_rng(seed).choice(plot_ys.size, size=n_plot, replace=False)
    ys = [plot_ys[plot_take]]
    xs = [plot_xs[plot_take]]
    is_plot = [np.ones(n_plot, np.uint8)]

    tiles = []
    for y0 in range(0, HEIGHT, TILE):
        for x0 in range(0, WIDTH, TILE):
            if np.any(veg[y0 : min(y0 + TILE, HEIGHT), x0 : min(x0 + TILE, WIDTH)]):
                tiles.append((y0, x0))
    pick = rng.choice(len(tiles), size=min(n_tiles, len(tiles)), replace=False)
    for y0, x0 in (tiles[i] for i in pick):
        y1, x1 = min(y0 + TILE, HEIGHT), min(x0 + TILE, WIDTH)
        local_y, local_x = np.nonzero(veg[y0:y1, x0:x1])
        take = rng.choice(local_y.size, size=min(per_tile, local_y.size), replace=False)
        ys.append(local_y[take].astype(np.int32) + y0)
        xs.append(local_x[take].astype(np.int32) + x0)
        is_plot.append(np.zeros(take.size, np.uint8))

    ys = np.concatenate(ys).astype(np.int32)
    xs = np.concatenate(xs).astype(np.int32)
    is_plot = np.concatenate(is_plot)
    # Drop metric pixels that duplicate the plot sample.
    key = ys.astype(np.int64) * WIDTH + xs.astype(np.int64)
    _, first = np.unique(key, return_index=True)
    keep = np.zeros(ys.size, np.bool_)
    keep[first] = True
    keep[:n_plot] = True
    ys, xs, is_plot = ys[keep], xs[keep], is_plot[keep]
    lon, lat = lonlat_of(ys, xs)
    log(f"reading {ys.size} pixels ({int(is_plot.sum())} plot)")

    raw = np.empty((ys.size, len(times), 6), np.int16)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (y, x) in enumerate(zip(ys, xs)):
        buckets.setdefault((int(y) // TILE * TILE, int(x) // TILE * TILE), []).append(i)
    with Dataset(SRC) as src:
        data = src.variables["data"]
        for n_done, ((y0, x0), idxs) in enumerate(buckets.items(), start=1):
            y1, x1 = min(y0 + TILE, HEIGHT), min(x0 + TILE, WIDTH)
            block = np.array(data[:, :, y0:y1, x0:x1], dtype=np.int16)
            rows = ys[idxs] - y0
            cols = xs[idxs] - x0
            raw[idxs] = block[:, :, rows, cols].transpose(2, 0, 1)
            if n_done % 8 == 0 or n_done == len(buckets):
                log(f"tiles {n_done}/{len(buckets)}")

    valid = np.all(raw > 0, axis=-1)
    holdout = np.zeros(valid.shape, np.uint8)
    for i in range(raw.shape[0]):
        choices = np.flatnonzero(valid[i])
        if choices.size == 0:
            continue
        n_hide = max(1, int(round(choices.size * HOLDOUT)))
        holdout[i, rng.choice(choices, size=min(n_hide, choices.size), replace=False)] = 1

    np.savez_compressed(
        OUT_DIR / "sample.npz",
        raw=raw,
        ys=ys,
        xs=xs,
        is_plot=is_plot,
        lon=lon,
        lat=lat,
        stamp=day_features(times),
        holdout=holdout,
        time_ns=times.asi8,
    )
    hk_mean, hk_std = hk_scaler()
    meta = {
        "n": int(ys.size),
        "n_plot": int(is_plot.sum()),
        "holdout_fraction": HOLDOUT,
        "bands_usa": list(BANDS6) + ["NDVI"],
        "usa_mean": USA_MEAN.tolist(),
        "usa_std": USA_STD.tolist(),
        "hk_mean": hk_mean.tolist(),
        "hk_std": hk_std.tolist(),
        "usa_checkpoint": str(USA_CKPT),
        "hk_checkpoint": str(HK_CKPT),
        "seed": seed,
    }
    (OUT_DIR / "sample_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {OUT_DIR / 'sample.npz'}")


def infer(which: str, batch_size: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("infer needs CUDA")
    device = torch.device("cuda:0")
    sample = np.load(OUT_DIR / "sample.npz")
    raw = sample["raw"]
    stamp = sample["stamp"]
    lon = sample["lon"]
    lat = sample["lat"]
    holdout = sample["holdout"].astype(bool)
    mean = USA_MEAN if which == "usa" else hk_scaler()[0]
    std = USA_STD if which == "usa" else hk_scaler()[1]
    model = load_model(which, device)
    log(f"{which} on {torch.cuda.get_device_name(0)} pixels {raw.shape[0]} batch {batch_size}")

    def run(series: np.ndarray) -> np.ndarray:
        parts = []
        for start in range(0, series.shape[0], batch_size):
            sl = slice(start, start + batch_size)
            values = with_ndvi(series[sl]) if which == "usa" else np.where(
                series[sl] > 0, series[sl].astype(np.float32), np.nan
            )
            scaled = scale(values, mean, std).astype(np.float32)
            pred = predict(model, scaled, stamp, lon[sl], lat[sl], device)
            restored = pred * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
            parts.append(restored[:, :, :6].astype(np.float32))
            log(f"{which} {sl.stop}/{series.shape[0]}")
        return np.concatenate(parts, axis=0)

    natural = run(raw)
    held = run(hide(raw, holdout))
    np.savez_compressed(OUT_DIR / f"pred_{which}.npz", natural=natural, holdout=held)
    log(f"wrote pred_{which}.npz")


def _reflectance(dn: np.ndarray) -> np.ndarray:
    return np.clip(dn.astype(np.float32) / 10000.0, 0.0, 1.0)


def _indices(cube: np.ndarray) -> dict[str, np.ndarray]:
    blue, red, nir = (_reflectance(cube[:, :, i]) for i in (0, 2, 3))
    nd = np.divide(nir - red, nir + red, out=np.full(red.shape, np.nan, np.float32), where=(nir + red) > 1e-6)
    evi_den = nir + 6.0 * red - 7.5 * blue + 1.0
    ev = np.divide(2.5 * (nir - red), evi_den, out=np.full(red.shape, np.nan, np.float32), where=np.abs(evi_den) > 1e-6)
    e2_den = nir + 2.4 * red + 1.0
    e2 = np.divide(2.5 * (nir - red), e2_den, out=np.full(red.shape, np.nan, np.float32), where=np.abs(e2_den) > 1e-6)
    return {"NDVI": nd, "EVI": ev, "EVI2": e2}


def _mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    err = np.abs(pred - truth)
    return float(err[mask].mean()) if np.any(mask) else float("nan")


def report() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    sample = np.load(OUT_DIR / "sample.npz")
    raw = sample["raw"].astype(np.float32)
    holdout = sample["holdout"].astype(bool)
    is_plot = sample["is_plot"].astype(bool)
    times = pd.to_datetime(sample["time_ns"])
    hk = np.load(OUT_DIR / "pred_hk.npz")
    usa = np.load(OUT_DIR / "pred_usa.npz")
    observed = raw > 0
    score = np.broadcast_to(holdout[:, :, None], raw.shape)

    def pack(pred: np.ndarray) -> dict:
        pred_ref = _reflectance(pred)
        raw_ref = _reflectance(raw)
        band = {
            name: {
                "mae_dn": _mae(pred[:, :, i], raw[:, :, i], score[:, :, i]),
                "mae_reflectance": _mae(pred_ref[:, :, i], raw_ref[:, :, i], score[:, :, i]),
            }
            for i, name in enumerate(BANDS6)
        }
        pred_idx = _indices(pred)
        raw_idx = _indices(raw)
        index = {}
        for name, need in {
            "NDVI": observed[:, :, 2] & observed[:, :, 3],
            "EVI": observed[:, :, 0] & observed[:, :, 2] & observed[:, :, 3],
            "EVI2": observed[:, :, 2] & observed[:, :, 3],
        }.items():
            mask = holdout & need
            index[name] = _mae(pred_idx[name], raw_idx[name], mask)
        per_pixel = np.abs(_indices(pred)["NDVI"] - _indices(raw)["NDVI"])
        per_pixel = np.where(holdout & observed[:, :, 2] & observed[:, :, 3], per_pixel, np.nan)
        return {"bands": band, "index_mae": index, "ndvi_mae_pixel": np.nanmean(per_pixel, axis=1)}

    hk_m = pack(hk["holdout"])
    usa_m = pack(usa["holdout"])
    both = np.isfinite(hk_m["ndvi_mae_pixel"]) & np.isfinite(usa_m["ndvi_mae_pixel"])
    usa_better = int(np.sum(usa_m["ndvi_mae_pixel"][both] < hk_m["ndvi_mae_pixel"][both]))
    summary = {
        "n_pixels": int(raw.shape[0]),
        "n_holdout_steps": int(holdout.sum()),
        "hk": {k: hk_m[k] for k in ("bands", "index_mae")},
        "usa": {k: usa_m[k] for k in ("bands", "index_mae")},
        "pixels_usa_lower_ndvi_mae": usa_better,
        "pixels_compared": int(both.sum()),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    fig_b, ax = plt.subplots(figsize=(8, 4.2))
    labels = list(BANDS6) + ["NDVI", "EVI", "EVI2"]
    hk_vals = [hk_m["bands"][name]["mae_reflectance"] for name in BANDS6] + [hk_m["index_mae"][n] for n in ("NDVI", "EVI", "EVI2")]
    usa_vals = [usa_m["bands"][name]["mae_reflectance"] for name in BANDS6] + [usa_m["index_mae"][n] for n in ("NDVI", "EVI", "EVI2")]
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.18, hk_vals, width=0.36, color="#1b7f4e", label="HK 6-band")
    ax.bar(xpos + 0.18, usa_vals, width=0.36, color="#d85a30", label="USA 7-band")
    ax.set_xticks(xpos, labels)
    ax.set_ylabel("Hold-out MAE")
    ax.set_title("25% observed steps hidden. Bands in reflectance, indices unitless.")
    ax.legend()
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    fig_b.tight_layout()
    bar_path = FIG_DIR / "compare_usa_hk_holdout_mae.png"
    fig_b.savefig(bar_path, dpi=140)
    plt.close(fig_b)
    log(f"wrote {bar_path}")

    plot_i = np.flatnonzero(is_plot)[:100]
    obs = _indices(np.where(observed[plot_i], raw[plot_i], np.nan))
    # Product view: keep real observations, draw the model only in gaps and as a line.
    hk_show = hk["natural"][plot_i].copy()
    usa_show = usa["natural"][plot_i].copy()
    hk_show[observed[plot_i]] = raw[plot_i][observed[plot_i]]
    usa_show[observed[plot_i]] = raw[plot_i][observed[plot_i]]
    rec_hk = _indices(hk_show)
    rec_usa = _indices(usa_show)
    names = ("NDVI", "EVI", "EVI2")
    n = plot_i.size
    ncol = 10
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(3 * nrow, ncol, figsize=(22, 3 * nrow * 1.15 + 1.6), sharex=True, squeeze=False)
    for block, name in enumerate(names):
        finite = np.concatenate([rec_hk[name][np.isfinite(rec_hk[name])], rec_usa[name][np.isfinite(rec_usa[name])]])
        lo, hi = np.nanpercentile(finite, [1, 99])
        pad = 0.08 * max(hi - lo, 0.2)
        for i in range(n):
            r, c = divmod(i, ncol)
            ax = axes[block * nrow + r, c]
            ax.plot(times, rec_hk[name][i], color="#1b7f4e", lw=1.0, label="HK" if i == 0 else None)
            ax.plot(times, rec_usa[name][i], color="#d85a30", lw=1.0, alpha=0.9, label="USA" if i == 0 else None)
            ax.scatter(times, obs[name][i], s=8, facecolors="white", edgecolors="black", linewidths=0.3, zorder=3)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(times[0], times[-1])
            ax.tick_params(labelsize=6, length=2)
            ax.grid(True, axis="y", lw=0.3, alpha=0.4)
            if r == 0 and c == ncol // 2:
                ax.set_title(name, fontsize=11)
            if c == 0 and r == nrow // 2:
                ax.set_ylabel(name, fontsize=8)
            ax.text(0.02, 0.92, f"{int(sample['ys'][plot_i][i])},{int(sample['xs'][plot_i][i])}", transform=ax.transAxes, fontsize=6, va="top")
            if block == 2 and r == nrow - 1:
                ax.xaxis.set_major_locator(mdates.YearLocator(2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            else:
                ax.tick_params(labelbottom=False)
        for i in range(n, nrow * ncol):
            r, c = divmod(i, ncol)
            axes[block * nrow + r, c].axis("off")
    fig.suptitle(
        "Hong Kong HLS 49QHE, 100 vegetation pixels\n"
        "Green: HK 6-band Imputator    Orange: USA 7-band Imputator (own z-score + NDVI)\n"
        "Open circles: original observations. Lines keep observations and fill gaps.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "compare_usa_hk_ndvi_evi_evi2.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log(f"wrote {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare USA and Hong Kong imputators on HLS 49QHE")
    p.add_argument("--stage", choices=("prepare", "infer", "report"), required=True)
    p.add_argument("--model", choices=("usa", "hk"))
    p.add_argument("--n-tiles", type=int, default=24)
    p.add_argument("--per-tile", type=int, default=96)
    p.add_argument("--n-plot", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--batch-size", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare(args.n_tiles, args.per_tile, args.n_plot, args.seed)
    elif args.stage == "infer":
        if args.model is None:
            raise SystemExit("--model is required for infer")
        infer(args.model, args.batch_size)
    else:
        report()


if __name__ == "__main__":
    main()
