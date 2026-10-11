#!/usr/bin/env python3
"""Show how a 3-day composite becomes a filled curve and then SOS/EOS.

Scene-level HLS granules are not on this machine. The earliest series stored
here is the cloud-screened 3-day median in veg_cube.nc. This figure follows
three vegetation pixels through cleaning, the sl122 fill, Savitzky–Golay
window 15, and the 15% / 50% thresholds.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from matplotlib import font_manager
from netCDF4 import Dataset

from phenology_hplm import evi
from phenology_threshold import STEPS, YEAR0, smooth_years, threshold_years

ROOT = Path(__file__).resolve().parent
PHENO = ROOT / "data" / "phenology"
OUT = ROOT / "figures" / "observation_to_phenology.png"
FONT = Path("/tmp/NotoSansSC-Regular.otf")
SIZE = 256
TILES = ((512, 256), (512, 768), (256, 1024))
LUM_NAME = {71: "林地", 72: "灌木", 73: "草地", 74: "红树林"}
TO_LL = pyproj.Transformer.from_crs(32649, 4326, always_xy=True)
X0, Y0, PIX = 799980.0, 2500020.0, 30.0


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def load_font():
    if not FONT.is_file():
        raise SystemExit(f"missing font {FONT}")
    font_manager.fontManager.addfont(str(FONT))
    return font_manager.FontProperties(fname=str(FONT))


def tile_evi(path: Path, y0: int, x0: int) -> np.ndarray:
    with Dataset(path) as src:
        data = np.array(src.variables["data"][:, :, y0:y0 + SIZE, x0:x0 + SIZE], np.int16)
    missing = np.all(data == 0, axis=1)
    values = evi(
        data[:, 0].astype(np.float32),
        data[:, 2].astype(np.float32),
        data[:, 3].astype(np.float32),
    )
    values[missing] = np.nan
    return values


def longest_gap(valid: np.ndarray) -> tuple[int, int]:
    best = cur = start = 0
    best_start = 0
    for i, flag in enumerate(valid):
        if flag:
            cur = 0
            continue
        if cur == 0:
            start = i
        cur += 1
        if cur > best:
            best = cur
            best_start = start
    return best, best_start


def doy_text(year: int, doy: float) -> str:
    day = datetime(year, 1, 1) + timedelta(days=int(round(doy)) - 1)
    return f"{day.month}月{day.day}日（第{int(round(doy))}天）"


def limb_levels(smooth: np.ndarray) -> dict | None:
    finite = np.isfinite(smooth)
    if int(finite.sum()) < 12:
        return None
    peak = int(np.nanargmax(np.where(finite, smooth, -1e9)))
    if peak < 4 or peak > STEPS - 5:
        return None
    left = int(np.nanargmin(np.where(finite[: peak + 1], smooth[: peak + 1], 1e9)))
    right = peak + int(np.nanargmin(np.where(finite[peak:], smooth[peak:], 1e9)))
    rise = float(smooth[peak] - smooth[left])
    fall = float(smooth[peak] - smooth[right])
    if rise < 0.04 or fall < 0.04:
        return None
    return {
        "peak": peak,
        "left": left,
        "right": right,
        "sos_level": float(smooth[left] + 0.15 * rise),
        "eos_level": float(smooth[peak] - 0.50 * fall),
        "rise": rise,
        "fall": fall,
    }


def collect() -> list[dict]:
    found = []
    doy = 1.0 + 3.0 * np.arange(STEPS)
    for y0, x0 in TILES:
        log(f"tile y={y0} x={x0}")
        with Dataset(PHENO / "veg_cube.nc") as src:
            veg = np.array(src.variables["vegetation"][y0:y0 + SIZE, x0:x0 + SIZE]) > 0
            lum = np.array(src.variables["lum_code"][y0:y0 + SIZE, x0:x0 + SIZE], np.int16)
        if int(veg.sum()) < 1000:
            continue
        raw = tile_evi(PHENO / "veg_cube.nc", y0, x0)
        clean = tile_evi(PHENO / "veg_cube_bolton.nc", y0, x0)
        filled = tile_evi(PHENO / "veg_filled_sl122.nc", y0, x0)
        ys, xs = np.nonzero(veg)
        raw_p = np.ascontiguousarray(raw[:, ys, xs].T)
        clean_p = np.ascontiguousarray(clean[:, ys, xs].T)
        filled_p = np.ascontiguousarray(filled[:, ys, xs].T)
        del raw, clean, filled
        log(f"  vegetation pixels {ys.size}")
        smooth = smooth_years(filled_p, 15)
        for year in range(2016, 2025):
            k = year - YEAR0
            sl = slice(k * STEPS, (k + 1) * STEPS)
            observed = np.isfinite(raw_p[:, sl])
            kept = np.isfinite(clean_p[:, sl])
            removed = observed & ~kept
            n_obs = observed.sum(axis=1)
            n_removed = removed.sum(axis=1)
            n_keep = kept.sum(axis=1)
            # Local excess of each removed point over the kept median nearby.
            spike = np.zeros(ys.size, np.float32)
            gap_len = np.zeros(ys.size, np.int16)
            gap_start = np.zeros(ys.size, np.int16)
            # Score a subset that already has enough observations and a real season.
            sos15, _, eos50, _ = threshold_years(np.ascontiguousarray(smooth[:, k:k + 1]))
            sos = sos15[:, 0]
            eos = eos50[:, 0]
            seasonal = (
                np.isfinite(sos) & np.isfinite(eos)
                & (sos >= 60) & (sos <= 140)
                & (eos >= 250) & (eos <= 340)
                & (n_obs >= 24) & (n_obs <= 70)
            )
            candidates = np.flatnonzero(seasonal & (n_removed >= 2))
            log(f"  {year} seasonal pixels with removals {candidates.size}")
            for i in candidates:
                extra = 0.0
                idx = np.flatnonzero(removed[i])
                series = clean_p[i, sl]
                for j in idx:
                    lo, hi = max(0, j - 4), min(STEPS, j + 5)
                    neigh = series[lo:hi]
                    neigh = neigh[np.isfinite(neigh)]
                    if neigh.size == 0:
                        continue
                    extra = max(extra, abs(float(raw_p[i, sl][j] - np.median(neigh))))
                spike[i] = extra
                length, start = longest_gap(kept[i])
                gap_len[i] = length
                gap_start[i] = start
            levels = []
            for i in candidates:
                info = limb_levels(smooth[i, k])
                if info is None:
                    continue
                gap_doys = doy[gap_start[i]:gap_start[i] + gap_len[i]]
                in_season = bool(np.any((gap_doys >= 60) & (gap_doys <= 300))) if gap_len[i] else False
                row = int(y0 + ys[i])
                col = int(x0 + xs[i])
                levels.append({
                    "row": row,
                    "col": col,
                    "year": year,
                    "lum": int(lum[ys[i], xs[i]]),
                    "n_obs": int(n_obs[i]),
                    "n_keep": int(n_keep[i]),
                    "n_removed": int(n_removed[i]),
                    "spike": float(spike[i]),
                    "gap": int(gap_len[i]),
                    "gap_start": int(gap_start[i]),
                    "gap_in_season": in_season,
                    "sos": float(sos[i]),
                    "eos": float(eos[i]),
                    "rise": info["rise"],
                    "raw": raw_p[i, sl].copy(),
                    "clean": clean_p[i, sl].copy(),
                    "filled": filled_p[i, sl].copy(),
                    "smooth": smooth[i, k].copy(),
                    "limb": info,
                })
            found.extend(levels)
        del raw_p, clean_p, filled_p, smooth
    log(f"candidates {len(found)}")
    return found


def apart(a: dict, b: dict, pixels: int = 80) -> bool:
    return abs(a["row"] - b["row"]) + abs(a["col"] - b["col"]) >= pixels


def within_scale(item: dict) -> bool:
    values = np.concatenate([item["raw"], item["filled"], item["smooth"]])
    values = values[np.isfinite(values)]
    return bool(values.size) and float(values.min()) > -0.15 and float(values.max()) < 1.05


def gap_is_bracketed(item: dict) -> bool:
    valid = np.isfinite(item["clean"])
    start = item["gap_start"]
    length = item["gap"]
    return int(valid[:start].sum()) >= 6 and int(valid[start + length:].sum()) >= 6


def choose(found: list[dict]) -> list[dict]:
    """One clear season, one with a visible spike removed, one with a filled gap."""
    usable = [item for item in found if within_scale(item)]
    classic = [
        item for item in usable
        if 3 <= item["n_removed"] <= 8
        and 5 <= item["gap"] <= 14
        and item["rise"] >= 0.18
        and item["n_obs"] >= 36
        and item["spike"] <= 0.40
    ]
    classic.sort(key=lambda item: abs(item["sos"] - 95) + abs(item["eos"] - 302))
    spiked = [
        item for item in usable
        if 0.30 <= item["spike"] <= 0.60
        and item["rise"] >= 0.18
        and item["n_obs"] >= 30
        and 1 <= item["n_removed"] <= 6
    ]
    spiked.sort(key=lambda item: abs(item["spike"] - 0.40))
    gappy = [
        item for item in usable
        if 12 <= item["gap"] <= 24
        and item["gap_in_season"]
        and item["rise"] >= 0.18
        and item["n_obs"] >= 30
        and gap_is_bracketed(item)
    ]
    gappy.sort(key=lambda item: abs(item["gap"] - 18))
    log(f"pools classic {len(classic)} spike {len(spiked)} gap {len(gappy)}")
    picked = []
    pools = (
        ("观测较完整，清洗只动了几个点", classic),
        ("清洗去掉了一个明显尖峰", spiked),
        ("生长季里有一段较长的缺口被填上", gappy),
    )
    for label, pool in pools:
        for item in pool:
            if any(not apart(item, prev) for prev in picked):
                continue
            if any(item["row"] == prev["row"] and item["col"] == prev["col"] for prev in picked):
                continue
            chosen = dict(item)
            chosen["story"] = label
            picked.append(chosen)
            break
    if len(picked) < 3:
        raise RuntimeError(f"only found {len(picked)} distinct examples")
    return picked


def check_against_product(examples: list[dict]) -> None:
    with Dataset(PHENO / "phenology_threshold_sl122_evi.nc") as src:
        row = np.array(src.variables["row"][:], np.int32)
        col = np.array(src.variables["col"][:], np.int32)
        sos = np.array(src.variables["sos15"][:], np.float32)
        eos = np.array(src.variables["eos50"][:], np.float32)
    key = row.astype(np.int64) * 1830 + col
    order = np.argsort(key)
    ordered = key[order]
    for item in examples:
        pos = int(np.searchsorted(ordered, item["row"] * 1830 + item["col"]))
        if pos >= ordered.size or ordered[pos] != item["row"] * 1830 + item["col"]:
            raise RuntimeError(f"pixel {item['row']},{item['col']} missing from phenology file")
        idx = int(order[pos])
        k = item["year"] - YEAR0
        stored_sos = float(sos[idx, k])
        stored_eos = float(eos[idx, k])
        if abs(stored_sos - item["sos"]) > 0.2 or abs(stored_eos - item["eos"]) > 0.2:
            raise RuntimeError(
                f"date mismatch at {item['row']},{item['col']} {item['year']}: "
                f"plot {item['sos']:.1f}/{item['eos']:.1f} file {stored_sos:.1f}/{stored_eos:.1f}"
            )
        log(f"matched product {item['row']},{item['col']} {item['year']}")


def draw(examples: list[dict], font) -> None:
    doy = 1.0 + 3.0 * np.arange(STEPS)
    month_doy = [1, 60, 121, 182, 244, 305]
    month_name = ["1月", "3月", "5月", "7月", "9月", "11月"]
    fig, axes = plt.subplots(4, 3, figsize=(14.8, 12.4), sharex=True, dpi=140)
    fig.subplots_adjust(left=0.175, right=0.985, top=0.78, bottom=0.11, hspace=0.48, wspace=0.20)
    row_names = [
        "1  三天合成后的观测",
        "2  清洗后留下的观测",
        "3  sl122 填成完整序列",
        "4  SG15 平滑后取返青、休眠",
    ]
    for col, item in enumerate(examples):
        raw = item["raw"]
        clean = item["clean"]
        filled = item["filled"]
        smooth = item["smooth"]
        limb = item["limb"]
        kept = np.isfinite(clean)
        removed = np.isfinite(raw) & ~kept
        imputed = ~np.isfinite(clean) & np.isfinite(filled)
        series = [raw, clean, filled, smooth]
        finite = np.concatenate([values[np.isfinite(values)] for values in series])
        lo = float(np.min(finite)) - 0.05
        hi = float(np.max(finite)) + 0.07
        if hi - lo < 0.5:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - 0.25, mid + 0.25
        east, north = X0 + (item["col"] + 0.5) * PIX, Y0 - (item["row"] + 0.5) * PIX
        lon, lat = TO_LL.transform(east, north)
        cover = LUM_NAME.get(item["lum"], f"代码{item['lum']}")
        axes[0, col].set_title(
            f"{cover}  {lat:.2f}°N, {lon:.2f}°E  {item['year']}年\n{item['story']}",
            fontproperties=font, fontsize=10, pad=6, color="#1a1a1a",
        )
        notes = (
            f"观测 {item['n_obs']} 步，缺失 {STEPS - item['n_obs']} 步",
            f"删去 {item['n_removed']} 步，留下 {item['n_keep']} 步",
            f"模型补上 {int(imputed.sum())} 步",
            "",
        )
        for row in range(4):
            ax = axes[row, col]
            ax.set_ylim(lo, hi)
            ax.set_xlim(1, 366)
            if item["gap"] >= 8:
                gap_end = min(STEPS, item["gap_start"] + item["gap"])
                ax.axvspan(doy[item["gap_start"]], doy[gap_end - 1] + 3, color="#f3f0c8", zorder=0)
            ax.axhline(0, color="#dddddd", lw=0.6, zorder=0)
            ax.grid(axis="y", color="#eeeeee", lw=0.6, zorder=0)
            if row == 0:
                ax.plot(doy[kept], raw[kept], "o", ms=3.2, color="#222222", label="保留的观测", zorder=3)
                if removed.any():
                    ax.plot(doy[removed], raw[removed], "x", ms=5.5, mew=1.4, color="#d62728", label="随后被删的观测", zorder=4)
            elif row == 1:
                ax.plot(doy[kept], clean[kept], "o", ms=3.2, color="#222222", zorder=3)
            elif row == 2:
                ax.plot(doy, filled, "-", color="#9ecae1", lw=1.0, zorder=2)
                ax.plot(doy[imputed], filled[imputed], "o", ms=2.6, color="#3182bd", label="模型填上的值", zorder=3)
                ax.plot(doy[kept], filled[kept], "o", ms=3.0, color="#222222", zorder=4)
            else:
                ax.plot(doy, filled, "-", color="#d6e6f5", lw=1.0, zorder=1)
                ax.plot(doy, smooth, "-", color="#e07a00", lw=2.0, label="SG15 平滑", zorder=3)
                ax.hlines(limb["sos_level"], doy[limb["left"]], doy[limb["peak"]], colors="#2ca02c", linestyles="--", lw=1.1, zorder=2)
                ax.hlines(limb["eos_level"], doy[limb["peak"]], doy[limb["right"]], colors="#6b3fa0", linestyles="--", lw=1.1, zorder=2)
                ax.axvline(item["sos"], color="#2ca02c", lw=1.2, label="返青 15%", zorder=2)
                ax.axvline(item["eos"], color="#6b3fa0", lw=1.2, label="休眠 50%", zorder=2)
                ax.plot(doy[limb["peak"]], smooth[limb["peak"]], "o", ms=4.5, color="#e07a00", zorder=4)
            if notes[row]:
                ax.text(
                    0.98, 0.96, notes[row], transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, fontproperties=font, color="#333333",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.88},
                )
            if col == 0:
                ax.set_ylabel("EVI", fontsize=10)
            if row == 3:
                ax.set_xticks(month_doy)
                ax.set_xticklabels(month_name, fontproperties=font, fontsize=9)
                ax.set_xlabel(
                    f"返青 {doy_text(item['year'], item['sos'])}    休眠 {doy_text(item['year'], item['eos'])}",
                    fontproperties=font, fontsize=8.5, labelpad=4,
                )
            else:
                ax.tick_params(labelbottom=False)
            ax.tick_params(labelsize=8)
    # Row labels sit to the left of each row, using the first column axis.
    for row, name in enumerate(row_names):
        axes[row, 0].annotate(
            name, xy=(0, 1.0), xycoords="axes fraction",
            xytext=(-8, 1), textcoords="offset points",
            fontproperties=font, fontsize=9.5, color="#24598a", fontweight="bold",
            annotation_clip=False, ha="right", va="bottom",
        )
    # The first annotation above duplicates row 0. Remove the extra title-like one.
    # It was drawn with a high y. Clear by not using it: redraw figure without that text.
    fig.texts.clear()
    fig.text(
        0.175, 0.975, "从三天合成观测到返青和休眠",
        fontproperties=font, fontsize=16, fontweight="bold", color="#1a1a1a", va="top",
    )
    fig.text(
        0.175, 0.938,
        "起点是云掩膜之后的 3 天中值。景级原始影像不在本机。黄底是该年最长的一段连续缺失。",
        fontproperties=font, fontsize=9.5, color="#444444", va="top",
    )
    fig.text(
        0.175, 0.912,
        "黑点是保留的观测，红叉是清洗删掉的点，蓝点是 sl122 填上的缺失步。橙线是 Savitzky–Golay 窗口 15。绿线、紫线是 15% 和 50% 阈值。",
        fontproperties=font, fontsize=9.5, color="#444444", va="top",
    )
    handles, labels = [], []
    for ax in axes.ravel():
        h, lab = ax.get_legend_handles_labels()
        for handle, label in zip(h, lab):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles, labels, loc="lower center", ncol=5, frameon=False,
        prop=font, fontsize=10, bbox_to_anchor=(0.52, 0.012),
    )
    # Drop the duplicate row-0 label that was added before fig.texts.clear did not remove annotations.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    log(f"wrote {OUT}")


def main() -> None:
    font = load_font()
    found = collect()
    examples = choose(found)
    check_against_product(examples)
    for item in examples:
        log(
            f"{item['story']} | {LUM_NAME.get(item['lum'], item['lum'])} "
            f"row {item['row']} col {item['col']} {item['year']} "
            f"obs {item['n_obs']} removed {item['n_removed']} gap {item['gap']} "
            f"SOS {item['sos']:.1f} EOS {item['eos']:.1f} spike {item['spike']:.2f}"
        )
    draw(examples, font)


if __name__ == "__main__":
    main()
