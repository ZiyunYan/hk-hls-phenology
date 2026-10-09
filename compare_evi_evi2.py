#!/usr/bin/env python3
"""Compare EVI and EVI2 as inputs to Zhang HPLM on both gap-filled cubes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from cube_io import TILE
from hk_paths import PHENO
from phenology_from_cube import fit_block
from phenology_hplm import N_YEARS, STEPS, greenness, log

HEIGHT = 1830
INDICES = ("evi", "evi2")


def load_pixels(path: Path, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    raw = np.empty((ys.size, N_YEARS * STEPS, 6), np.int16)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (y, x) in enumerate(zip(ys, xs)):
        buckets.setdefault((int(y) // TILE * TILE, int(x) // TILE * TILE), []).append(i)
    with Dataset(path) as src:
        data = src.variables["data"]
        n_time = data.shape[0]
        if n_time != N_YEARS * STEPS:
            raise SystemExit(f"{path.name} has {n_time} steps, expected {N_YEARS * STEPS}")
        for n_done, ((y0, x0), idxs) in enumerate(buckets.items(), start=1):
            y1, x1 = min(y0 + TILE, HEIGHT), min(x0 + TILE, HEIGHT)
            tile = np.array(data[:, :, y0:y1, x0:x1], np.int16)
            rows = ys[idxs] - y0
            cols = xs[idxs] - x0
            raw[idxs] = tile[:, :, rows, cols].transpose(2, 0, 1)
            if n_done % 8 == 0 or n_done == len(buckets):
                log(f"{path.name} tiles {n_done}/{len(buckets)}")
    return raw


def sample_pixels(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(PHENO / "veg_filled.nc") as src:
        veg = np.array(src.variables["vegetation"][:], np.uint8) > 0
    ys, xs = np.nonzero(veg)
    take = np.random.default_rng(seed).choice(ys.size, n, replace=False)
    return ys[take].astype(np.int32), xs[take].astype(np.int32)


def _curve_stats(series: np.ndarray) -> dict[str, float]:
    years = series.reshape(series.shape[0], N_YEARS, STEPS)
    amp = np.nanmax(years, axis=-1) - np.nanmin(years, axis=-1)
    d2 = np.mean(np.abs(np.diff(years, n=2, axis=-1)), axis=-1)
    ratio = np.divide(d2, amp, out=np.full(amp.shape, np.nan), where=amp > 0.02)
    return {
        "median_amplitude": float(np.nanmedian(amp)),
        "median_roughness_over_amp": float(np.nanmedian(ratio)),
    }


def _metric_stats(sos: np.ndarray, eos: np.ndarray) -> dict[str, float]:
    both = np.isfinite(sos) & np.isfinite(eos)
    season = np.where(both, eos - sos, np.nan)
    count = np.sum(np.isfinite(sos), axis=1)
    keep = count >= 5
    sos_std = np.nanstd(sos[keep], axis=1) if np.any(keep) else np.empty(0)
    return {
        "valid_pairs": int(both.sum()),
        "valid_pair_fraction": float(both.mean()),
        "median_sos": float(np.nanmedian(sos)),
        "sos_p25": float(np.nanpercentile(sos[np.isfinite(sos)], 25)),
        "sos_p75": float(np.nanpercentile(sos[np.isfinite(sos)], 75)),
        "median_eos": float(np.nanmedian(eos)),
        "eos_p25": float(np.nanpercentile(eos[np.isfinite(eos)], 25)),
        "eos_p75": float(np.nanpercentile(eos[np.isfinite(eos)], 75)),
        "median_season_days": float(np.nanmedian(season)),
        "median_sos_std": float(np.median(sos_std)) if sos_std.size else float("nan"),
    }


def agreement(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(a) & np.isfinite(b)
    delta = np.abs(a[ok] - b[ok])
    if delta.size == 0:
        return {"n": 0, "median_abs_days": float("nan")}
    return {"n": int(delta.size), "median_abs_days": float(np.median(delta))}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--smooth", default="sg9")
    args = p.parse_args()
    ys, xs = sample_pixels(args.n, args.seed)
    cubes = {
        "hk": PHENO / "veg_filled.nc",
        "usa": PHENO / "veg_filled_usa.nc",
    }
    fitted: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    curves: dict[tuple[str, str], np.ndarray] = {}
    summary: dict[str, dict] = {}
    for tag, path in cubes.items():
        raw = load_pixels(path, ys, xs)
        summary[tag] = {}
        for index in INDICES:
            series = greenness(raw, index)
            log(f"fit {tag} {index}")
            sos, eos = fit_block(series, args.smooth)
            fitted[(tag, index)] = (sos, eos)
            curves[(tag, index)] = series
            summary[tag][index] = {**_curve_stats(series), **_metric_stats(sos, eos)}
            log(f"{tag} {index} {json.dumps(summary[tag][index])}")
    summary["agreement_sos"] = {
        "evi_hk_vs_usa": agreement(fitted[("hk", "evi")][0], fitted[("usa", "evi")][0]),
        "evi2_hk_vs_usa": agreement(fitted[("hk", "evi2")][0], fitted[("usa", "evi2")][0]),
        "hk_evi_vs_evi2": agreement(fitted[("hk", "evi")][0], fitted[("hk", "evi2")][0]),
        "usa_evi_vs_evi2": agreement(fitted[("usa", "evi")][0], fitted[("usa", "evi2")][0]),
    }
    out = PHENO / "evi_evi2_compare.json"
    out.write_text(json.dumps({"n_pixels": int(ys.size), "smooth": args.smooth, "summary": summary}, indent=2))
    log(f"wrote {out}")
    _plot(fitted, summary, ys, xs)
    np.savez_compressed(
        PHENO / "evi_evi2_compare_sample.npz",
        ys=ys,
        xs=xs,
        **{f"sos_{tag}_{index}": fitted[(tag, index)][0] for tag in cubes for index in INDICES},
        **{f"eos_{tag}_{index}": fitted[(tag, index)][1] for tag in cubes for index in INDICES},
    )


def _plot(fitted, summary, ys, xs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    labels = ["HK EVI", "HK EVI2", "USA EVI", "USA EVI2"]
    keys = [("hk", "evi"), ("hk", "evi2"), ("usa", "evi"), ("usa", "evi2")]
    colors = ["#1b7f4e", "#7dba6a", "#d85a30", "#e39a78"]
    vals = [summary[tag][index]["valid_pair_fraction"] * 100 for tag, index in keys]
    axes[0].bar(labels, vals, color=colors)
    axes[0].set_ylabel("Valid SOS+EOS (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Zhang HPLM success, SG-9")
    axes[0].grid(True, axis="y", lw=0.3, alpha=0.4)
    bins = np.linspace(1, 366, 37)
    for (tag, index), color, label in zip(keys, colors, labels):
        sos = fitted[(tag, index)][0]
        axes[1].hist(sos[np.isfinite(sos)], bins=bins, histtype="step", lw=1.6, color=color, label=label, density=True)
    axes[1].set_xlabel("SOS day of year")
    axes[1].set_ylabel("Density")
    axes[1].set_title("SOS timing")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "compare_evi_evi2_phenology.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
