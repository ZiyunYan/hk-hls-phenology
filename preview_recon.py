#!/usr/bin/env python3
"""Preview the smooth-beta-1 top-k-5 imputator on held-out Hong Kong pixels."""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset

SSL = Path(__file__).resolve().parent / "imputator_ssl"
sys.path.insert(0, str(SSL))
from models.Transformer import Model  # noqa: E402
from utils.timefeatures import time_features  # noqa: E402

SRC = Path("/intelnvme03/ziyun218/hls_49QHE_hk/imputator_hk/test.nc")
SCALER = Path("/intelnvme03/ziyun218/hls_49QHE_hk/imputator_hk/train.nc")
CKPT = (
    SSL
    / "checkpoints/HK-Imputator-optical6-topk5-s1-sl366/checkpoint.pth"
)
OUT = Path("/home/ziyun218/pyprojects/Transformer4Imputation/plots/hk_recon_topk5_s1.png")
SEQ, STRIDE, N_YEARS = 366, 122, 11
NAMES = ["Blue", "Red", "NIR", "SWIR1"]
BANDS = [0, 2, 3, 4]
YEAR = 5  # 2020, interior year so the plotted window uses its middle third


def configs() -> Namespace:
    return Namespace(
        enc_in=6, d_model=256, n_heads=8, e_layers=6, d_ff=1024,
        embed="timeF", freq="rs", dropout=0.0, factor=1, output_attention=True,
        activation="gelu", mask_rate=0.8, imp_n_storage_tokens=2,
        lon_lat_n_fourier_freqs=4, use_lon_lat_embed=1, geo_dropout_p=0.5,
        imp_rec_loss="mse", imp_huber_delta=1.0, imp_rec_alpha=1.0,
        imp_smooth_beta=1.0, imp_smooth_mode="dy2",
        imp_trim_topk_per_seq=5, imp_trim_min_keep=8, imp_mask_min_p=0.4,
    )


def year_window(year_index: int) -> tuple[int, slice]:
    if year_index == 0:
        return 0, slice(0, STRIDE)
    if year_index == N_YEARS - 1:
        return (N_YEARS - 3) * STRIDE, slice(2 * STRIDE, SEQ)
    return (year_index - 1) * STRIDE, slice(STRIDE, 2 * STRIDE)


def predict_window(model, window, stamp, lon, lat):
    take = window.shape[0]
    mark = np.broadcast_to(stamp, (take, SEQ, stamp.shape[-1])).copy()
    ll = np.stack([lon, lat], axis=-1)
    llw = np.broadcast_to(ll[:, None, :], (take, SEQ, 2)).copy()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(window).cuda(),
            time_mark=torch.from_numpy(mark).cuda(),
            lon_lat=torch.from_numpy(llw).cuda(),
            mode="pred",
        )
    return pred.float().cpu().numpy()


def seasonal_picks(nir: np.ndarray) -> list[int]:
    """Prefer vegetation-like series: seasonal NIR, not a few bright outliers."""
    finite = np.isfinite(nir)
    cover = finite.mean(axis=1)
    med = np.nanmedian(nir, axis=1)
    p95 = np.nanpercentile(np.where(finite, nir, np.nan), 95, axis=1)
    # Rough annual contrast: upper quartile minus lower quartile.
    q75 = np.nanpercentile(nir, 75, axis=1)
    q25 = np.nanpercentile(nir, 25, axis=1)
    contrast = q75 - q25
    ok = (cover > 0.12) & (cover < 0.55) & (med > 1500) & (med < 4500) & (p95 < 6500) & (contrast > 400)
    score = np.where(ok, contrast, -1)
    order = np.argsort(score)[::-1]
    return [int(i) for i in order[:4] if score[i] > 0]


def main() -> None:
    with Dataset(SCALER) as f:
        mean = np.array(f.variables["band_mean"][:], np.float32)
        std = np.array(f.variables["band_std"][:], np.float32)
    model = Model(configs())
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval().cuda()

    start, _ = year_window(YEAR)
    with Dataset(SRC) as src:
        times = [str(t) for t in src.variables["time"][:]]
        stamp = time_features(pd.to_datetime(times, format="%Y%j"), freq="rs").T.astype(np.float32)
        n = len(src.dimensions["pixels"])
        scan_n = min(1500, n)
        nir = np.array(src.variables["data"][:scan_n, :, 3], np.float32)
        picks = seasonal_picks(nir)
        if len(picks) < 4:
            raise SystemExit(f"only {len(picks)} seasonal pixels in the scan")
        raw = np.array(src.variables["data"][picks, start:start + SEQ], np.float32)
        lon = np.array(src.variables["lon"][picks], np.float32)
        lat = np.array(src.variables["lat"][picks], np.float32)
        metric_n = min(256, n)
        metric_raw = np.array(src.variables["data"][:metric_n, start:start + SEQ], np.float32)
        metric_lon = np.array(src.variables["lon"][:metric_n], np.float32)
        metric_lat = np.array(src.variables["lat"][:metric_n], np.float32)

    x = (raw - mean) / std
    pred = predict_window(model, x, stamp[start:start + SEQ], lon, lat)
    recon = pred * std + mean
    seen = np.isfinite(raw)
    mae = np.abs(recon - raw)[seen].mean()
    print(f"window {times[start]}-{times[start + SEQ - 1]} observed MAE {mae:.2f} DN on {int(seen.sum())} points, {len(picks)} pixels")

    mx = (metric_raw - mean) / std
    mpred = predict_window(model, mx, stamp[start:start + SEQ], metric_lon, metric_lat) * std + mean
    mseen = np.isfinite(metric_raw)
    mmae = np.abs(mpred - metric_raw)[mseen].mean()
    print(f"same window, first {metric_n} test pixels, observed MAE {mmae:.2f} DN, points {int(mseen.sum())}")
    for band, name in zip(BANDS, NAMES):
        band_seen = mseen[:, :, band]
        band_mae = np.abs(mpred[:, :, band] - metric_raw[:, :, band])[band_seen].mean()
        print(f"  {name} {band_mae:.2f} DN, points {int(band_seen.sum())}")

    steps = np.arange(SEQ)
    fig, axes = plt.subplots(4, 4, figsize=(16, 10), sharex=True)
    for row in range(4):
        for col, (band, name) in enumerate(zip(BANDS, NAMES)):
            ax = axes[row, col]
            y = raw[row, :, band]
            yhat = recon[row, :, band]
            ok = np.isfinite(y)
            ax.plot(steps, yhat, color="#e67e22", lw=1.0, label="reconstructed")
            ax.scatter(steps[ok], y[ok], s=10, c="#1f4e79", label="observed", zorder=3)
            ax.axvspan(STRIDE, 2 * STRIDE, color="#f4e4c4", zorder=0)
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(f"{lat[row]:.3f}N {lon[row]:.3f}E\nDN")
            ax.grid(True, lw=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("3-day step in the 2019-2021 window (shade is 2020)")
    fig.suptitle(
        f"beta=1, top-k=5, pred mode. Observed MAE {mmae:.0f} DN on 256 test pixels",
        fontsize=12,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
