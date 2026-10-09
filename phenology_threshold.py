#!/usr/bin/env python3
"""15% and 50% amplitude thresholds on gap-filled EVI.

Each calendar year is split at the smoothed EVI peak. Rising dates are the
first crossing of 15% and 50% of the rise from the pre-peak minimum.
Declining dates are the first crossing of 50% and 15% of the drop toward
the post-peak minimum. Crossings are linearly interpolated between the
3-day samples.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from numba import njit, prange
from scipy.signal import savgol_filter

from phenology_hplm import N_YEARS, STEPS, YEAR0, greenness

MIN_AMP = 0.04
MIN_VALID = 12
MIN_SEASON = 30.0
HEIGHT = 1830


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def smooth_years(evi: np.ndarray, window: int) -> np.ndarray:
    """evi (pixels, time) -> smoothed (pixels, years, 122)."""
    years = np.ascontiguousarray(evi, np.float32).reshape(-1, N_YEARS, STEPS)
    flat = years.reshape(-1, STEPS).astype(np.float64, copy=True)
    bad = ~np.isfinite(flat)
    if bad.any():
        with np.errstate(all="ignore"):
            row_med = np.nanmedian(flat, axis=1)
        row_med = np.where(np.isfinite(row_med), row_med, 0.0)
        flat = np.where(bad, row_med[:, None], flat)
    sm = savgol_filter(flat, window_length=window, polyorder=2, axis=-1, mode="interp")
    sm[bad] = np.nan
    return sm.reshape(years.shape).astype(np.float32)


@njit
def _cross(t, y, i0, i1, level, rising):
    prev = -1
    for i in range(i0, i1 + 1):
        if not np.isfinite(y[i]):
            prev = -1
            continue
        if prev >= 0:
            y0 = y[prev]
            y1 = y[i]
            hit = (y0 < level <= y1) if rising else (y0 > level >= y1)
            if hit:
                den = y1 - y0
                frac = 0.0 if den == 0.0 else (level - y0) / den
                return t[prev] + frac * (t[i] - t[prev])
        prev = i
    return np.nan


@njit
def threshold_one(smooth):
    """Return sos15, sos50, eos50, eos15. NaN when that crossing is missing."""
    t = np.empty(STEPS, np.float64)
    for i in range(STEPS):
        t[i] = 1.0 + 3.0 * i
    nfin = 0
    for i in range(STEPS):
        if np.isfinite(smooth[i]):
            nfin += 1
    if nfin < MIN_VALID:
        return np.nan, np.nan, np.nan, np.nan
    peak = -1
    peak_v = -1e9
    for i in range(STEPS):
        if np.isfinite(smooth[i]) and smooth[i] > peak_v:
            peak_v = smooth[i]
            peak = i
    if peak < 4 or peak > STEPS - 5:
        return np.nan, np.nan, np.nan, np.nan
    left = 0
    left_v = smooth[peak]
    for i in range(peak):
        if np.isfinite(smooth[i]) and smooth[i] <= left_v:
            left_v = smooth[i]
            left = i
    right = STEPS - 1
    right_v = smooth[peak]
    for i in range(peak, STEPS):
        if np.isfinite(smooth[i]) and smooth[i] <= right_v:
            right_v = smooth[i]
            right = i
    rise = peak_v - left_v
    fall = peak_v - right_v
    sos15 = sos50 = eos50 = eos15 = np.nan
    if rise >= MIN_AMP and peak - left >= 3:
        sos15 = _cross(t, smooth, left, peak, left_v + 0.15 * rise, True)
        sos50 = _cross(t, smooth, left, peak, left_v + 0.50 * rise, True)
    if fall >= MIN_AMP and right - peak >= 3:
        eos50 = _cross(t, smooth, peak, right, peak_v - 0.50 * fall, False)
        eos15 = _cross(t, smooth, peak, right, right_v + 0.15 * fall, False)
    if np.isfinite(sos15) and np.isfinite(eos15) and eos15 <= sos15 + MIN_SEASON:
        sos15 = np.nan
        eos15 = np.nan
    if np.isfinite(sos50) and np.isfinite(eos50) and eos50 <= sos50 + MIN_SEASON:
        sos50 = np.nan
        eos50 = np.nan
    return sos15, sos50, eos50, eos15


@njit(parallel=True)
def threshold_years(smooth):
    """smooth (pixels, years, 122) -> four (pixels, years) date arrays."""
    n = smooth.shape[0]
    nyears = smooth.shape[1]
    sos15 = np.empty((n, nyears), np.float32)
    sos50 = np.empty((n, nyears), np.float32)
    eos50 = np.empty((n, nyears), np.float32)
    eos15 = np.empty((n, nyears), np.float32)
    for i in prange(n):
        for year in range(nyears):
            a, b, c, d = threshold_one(smooth[i, year])
            sos15[i, year] = a
            sos50[i, year] = b
            eos50[i, year] = c
            eos15[i, year] = d
    return sos15, sos50, eos50, eos15


def extra_crossings(smooth: np.ndarray, dates: tuple[np.ndarray, ...]) -> None:
    """Share of valid limbs that cross 15% more than once. Diagnostic only."""
    sos15, sos50, eos50, eos15 = dates
    up = down = up_multi = down_multi = 0
    # A few thousand seasons is enough.
    rng = np.random.default_rng(0)
    pix = np.arange(smooth.shape[0])
    if pix.size > 800:
        pix = rng.choice(pix, 800, replace=False)
    doy = 1.0 + 3.0 * np.arange(STEPS)
    for i in pix:
        for year in range(N_YEARS):
            if not np.isfinite(sos15[i, year]):
                continue
            y = smooth[i, year]
            finite = np.isfinite(y)
            p = int(np.nanargmax(np.where(finite, y, -1e9)))
            left = int(np.nanargmin(np.where(finite[: p + 1], y[: p + 1], 1e9)))
            right = p + int(np.nanargmin(np.where(finite[p:], y[p:], 1e9)))
            rise = float(y[p] - y[left])
            fall = float(y[p] - y[right])
            if rise >= MIN_AMP:
                level = y[left] + 0.15 * rise
                seg = y[left : p + 1]
                below = seg < level
                flips = int(np.sum((~below[1:]) & below[:-1]))
                up += 1
                up_multi += int(flips > 1)
            if fall >= MIN_AMP and np.isfinite(eos15[i, year]):
                level = y[right] + 0.15 * fall
                seg = y[p : right + 1]
                above = seg > level
                flips = int(np.sum((~above[1:]) & above[:-1]))
                down += 1
                down_multi += int(flips > 1)
    if up:
        print(f"  15% rise crossed more than once: {up_multi / up:.3f} of {up}")
    if down:
        print(f"  15% drop crossed more than once: {down_multi / down:.3f} of {down}")
    del sos50, eos50, doy


def summarize(name: str, dates: dict[str, np.ndarray]) -> dict[str, float]:
    print(name)
    med = {}
    nyears = next(iter(dates.values())).size
    for key, vals in dates.items():
        ok = np.isfinite(vals)
        q = np.nanpercentile(vals, [25, 50, 75]) if ok.any() else (np.nan, np.nan, np.nan)
        med[key] = float(q[1])
        print(f"  {key:6} {ok.mean() * 100:5.1f}%  p50={q[1]:6.1f}  p25={q[0]:6.1f}  p75={q[2]:6.1f}")
    both15 = np.isfinite(dates["sos15"]) & np.isfinite(dates["eos15"])
    both50 = np.isfinite(dates["sos50"]) & np.isfinite(dates["eos50"])
    print(f"  pair15 {both15.mean() * 100:5.1f}%   pair50 {both50.mean() * 100:5.1f}%   n={nyears}")
    return med


def compare_windows(src: Path, windows: list[int]) -> None:
    rng = np.random.default_rng(0)
    tiles = [(512, 768), (256, 1280), (640, 1024)]
    blocks = []
    with Dataset(src) as f:
        veg = np.array(f.variables["vegetation"][:]) > 0
        for y0, x0 in tiles:
            local = veg[y0 : y0 + 128, x0 : x0 + 128]
            rows, cols = np.nonzero(local)
            take = rng.choice(rows.size, min(400, rows.size), replace=False)
            raw = np.array(f.variables["data"][:, :, y0 : y0 + 128, x0 : x0 + 128], np.int16)
            blocks.append(raw[:, :, rows[take], cols[take]].transpose(2, 0, 1))
            print(y0, x0, take.size, flush=True)
    evi = greenness(np.concatenate(blocks, 0), "evi")
    print("pixels", evi.shape[0], flush=True)
    meds = []
    for window in windows:
        sm = smooth_years(evi, window)
        sos15, sos50, eos50, eos15 = threshold_years(np.ascontiguousarray(sm))
        packed = {
            "sos15": sos15.ravel(),
            "sos50": sos50.ravel(),
            "eos50": eos50.ravel(),
            "eos15": eos15.ravel(),
        }
        meds.append(summarize(f"sg{window}", packed))
        extra_crossings(sm, (sos15, sos50, eos50, eos15))
    base = meds[0]
    for window, med in zip(windows[1:], meds[1:]):
        shifts = {k: med[k] - base[k] for k in base}
        text = "  ".join(f"{k} {shifts[k]:+.1f}d" for k in shifts)
        print(f"sg{window} minus sg{windows[0]}: {text}")


def extract(src: Path, dst: Path, window: int) -> None:
    t0 = time.time()
    with Dataset(src) as f:
        veg = np.array(f.variables["vegetation"][:], np.uint8) > 0
        n = int(veg.sum())
        log(f"vegetation pixels {n} window={window}")
        sos15 = np.full((n, N_YEARS), np.nan, np.float32)
        sos50 = np.full((n, N_YEARS), np.nan, np.float32)
        eos50 = np.full((n, N_YEARS), np.nan, np.float32)
        eos15 = np.full((n, N_YEARS), np.nan, np.float32)
        cursor = 0
        data = f.variables["data"]
        strip = 24
        for y0 in range(0, HEIGHT, strip):
            y1 = min(y0 + strip, HEIGHT)
            local = veg[y0:y1]
            rows, cols = np.nonzero(local)
            if rows.size == 0:
                continue
            raw = np.array(data[:, :, y0:y1, :], np.int16)
            pix = raw[:, :, rows, cols].transpose(2, 0, 1)
            del raw
            sm = smooth_years(greenness(pix, "evi"), window)
            a, b, c, d = threshold_years(np.ascontiguousarray(sm))
            nstrip = rows.size
            sos15[cursor : cursor + nstrip] = a
            sos50[cursor : cursor + nstrip] = b
            eos50[cursor : cursor + nstrip] = c
            eos15[cursor : cursor + nstrip] = d
            cursor += nstrip
            log(f"rows {y1}/{HEIGHT} pixels {cursor}/{n} ({time.time() - t0:.0f}s)")
        if cursor != n:
            raise RuntimeError(f"pixel count {cursor} != vegetation {n}")
        row, col = np.nonzero(veg)
        if dst.exists():
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(dst, "w", format="NETCDF4") as out:
            out.createDimension("pixels", n)
            out.createDimension("year", N_YEARS)
            out.createVariable("year", "i2", ("year",))[:] = np.arange(YEAR0, YEAR0 + N_YEARS, dtype=np.int16)
            out.createVariable("row", "i4", ("pixels",))[:] = row.astype(np.int32)
            out.createVariable("col", "i4", ("pixels",))[:] = col.astype(np.int32)
            out.createVariable("sos15", "f4", ("pixels", "year"), zlib=True, complevel=1)[:] = sos15
            out.createVariable("sos50", "f4", ("pixels", "year"), zlib=True, complevel=1)[:] = sos50
            out.createVariable("eos50", "f4", ("pixels", "year"), zlib=True, complevel=1)[:] = eos50
            out.createVariable("eos15", "f4", ("pixels", "year"), zlib=True, complevel=1)[:] = eos15
            out.setncattr("index", "evi")
            out.setncattr("method", "amplitude threshold, 15% and 50% of each limb")
            out.setncattr("smooth", f"savgol window {window}, polyorder 2")
            out.setncattr("source_cube", str(src))
            out.setncattr("min_amplitude", MIN_AMP)
            both15 = np.isfinite(sos15) & np.isfinite(eos15)
            both50 = np.isfinite(sos50) & np.isfinite(eos50)
            out.setncattr("n_valid_15", int(both15.sum()))
            out.setncattr("n_valid_50", int(both50.sum()))
            log(f"pair 15% {int(both15.sum())} ({both15.mean() * 100:.1f}%)")
            log(f"pair 50% {int(both50.sum())} ({both50.mean() * 100:.1f}%)")
    log(f"wrote {dst} in {time.time() - t0:.0f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("data/phenology/veg_filled_sl122.nc"))
    p.add_argument("--dst", type=Path, default=Path("data/phenology/phenology_threshold_sl122_evi.nc"))
    p.add_argument("--window", type=int, default=15)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--windows", type=int, nargs="*", default=[9, 15, 21])
    args = p.parse_args()
    if args.window % 2 == 0 or args.window < 5:
        raise SystemExit("Savitzky–Golay window must be odd and at least 5")
    if args.compare:
        compare_windows(args.src, args.windows)
    else:
        extract(args.src, args.dst, args.window)


if __name__ == "__main__":
    main()
