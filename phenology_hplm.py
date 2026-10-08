#!/usr/bin/env python3
"""Annual SOS and EOS from gap-filled EVI2.

The fit is Xiaoyang Zhang's hybrid piecewise logistic model. Each year is split
at the greenness peak. An increasing logistic is fit to the green-up limb and a
decreasing logistic to the senescence limb. SOS is the green-up onset and EOS is
the dormancy onset, both at the curvature-change extrema
t0 ± ln(2+sqrt(3))/k. Pixel row, column, longitude and latitude are copied.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from numba import njit

ROOT = Path("/intelnvme03/ziyun218/hls_49QHE_hk/phenology")
SRC = ROOT / "veg_filled.nc"
DST = ROOT / "phenology_sos_eos.nc"
STEPS = 122
N_YEARS = 11
YEAR0 = 2015
DELTA = float(np.log(2.0 + np.sqrt(3.0)))
MIN_AMP = 0.04
MIN_VALID = 12


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def evi2(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    red_r = np.clip(red / 10000.0, 0.0, 1.0)
    nir_r = np.clip(nir / 10000.0, 0.0, 1.0)
    return (2.5 * (nir_r - red_r) / (nir_r + 2.4 * red_r + 1.0)).astype(np.float32)


@njit
def _sigmoid(z):
    if z > 40.0:
        return 1.0
    if z < -40.0:
        return 0.0
    return 1.0 / (1.0 + np.exp(-z))


@njit
def _fit_logistic(t, y, increasing):
    """Gauss-Newton fit of vmin + amp * sigmoid(sign * k * (t - t0)).

    Returns t0, k, amplitude. k is positive. NaN t0 means the fit failed.
    """
    n = t.shape[0]
    if n < 8:
        return np.nan, np.nan, np.nan
    vmin = y[0]
    vmax = y[0]
    for i in range(n):
        if y[i] < vmin:
            vmin = y[i]
        if y[i] > vmax:
            vmax = y[i]
    amp = vmax - vmin
    if amp < MIN_AMP:
        return np.nan, np.nan, np.nan
    mid = 0.5 * (vmin + vmax)
    t0 = t[n // 2]
    for i in range(n - 1):
        if increasing and y[i] <= mid <= y[i + 1]:
            t0 = 0.5 * (t[i] + t[i + 1])
            break
        if (not increasing) and y[i] >= mid >= y[i + 1]:
            t0 = 0.5 * (t[i] + t[i + 1])
            break
    sign = 1.0 if increasing else -1.0
    k = 0.08
    base = vmin
    for _ in range(25):
        jt0 = np.zeros(n)
        jk = np.zeros(n)
        jamp = np.zeros(n)
        jb = np.zeros(n)
        resid = np.zeros(n)
        for i in range(n):
            z = sign * k * (t[i] - t0)
            s = _sigmoid(z)
            pred = base + amp * s
            resid[i] = y[i] - pred
            ds = s * (1.0 - s)
            jt0[i] = amp * ds * (-sign * k)
            jk[i] = amp * ds * (sign * (t[i] - t0))
            jamp[i] = s
            jb[i] = 1.0
        # normal equations 4x4
        j = (jt0, jk, jamp, jb)
        a = np.zeros((4, 4))
        b = np.zeros(4)
        for r in range(4):
            for c in range(4):
                acc = 0.0
                for i in range(n):
                    acc += j[r][i] * j[c][i]
                a[r, c] = acc
            acc = 0.0
            for i in range(n):
                acc += j[r][i] * resid[i]
            b[r] = acc
        a[0, 0] += 1e-4
        a[1, 1] += 1e-4
        a[2, 2] += 1e-6
        a[3, 3] += 1e-6
        # Gaussian elimination
        for col in range(4):
            piv = col
            for r in range(col + 1, 4):
                if abs(a[r, col]) > abs(a[piv, col]):
                    piv = r
            if abs(a[piv, col]) < 1e-12:
                return np.nan, np.nan, np.nan
            if piv != col:
                for c in range(4):
                    tmp = a[col, c]
                    a[col, c] = a[piv, c]
                    a[piv, c] = tmp
                tmpb = b[col]
                b[col] = b[piv]
                b[piv] = tmpb
            div = a[col, col]
            for c in range(col, 4):
                a[col, c] /= div
            b[col] /= div
            for r in range(4):
                if r == col:
                    continue
                factor = a[r, col]
                for c in range(col, 4):
                    a[r, c] -= factor * a[col, c]
                b[r] -= factor * b[col]
        t0 += b[0]
        k += b[1]
        amp += b[2]
        base += b[3]
        if k < 0.005:
            k = 0.005
        if k > 1.0:
            k = 1.0
        if amp < 0.0:
            amp = 0.0
        if t0 < t[0]:
            t0 = t[0]
        if t0 > t[n - 1]:
            t0 = t[n - 1]
    if amp < MIN_AMP:
        return np.nan, np.nan, np.nan
    return t0, k, amp


@njit
def metrics_one_year(y):
    """y is 122 EVI2 samples at DOY 1,4,...,364. Returns SOS, EOS as DOY."""
    t = np.empty(122, np.float64)
    for i in range(122):
        t[i] = 1.0 + 3.0 * i
    finite = np.zeros(122, np.uint8)
    nfin = 0
    for i in range(122):
        if np.isfinite(y[i]):
            finite[i] = 1
            nfin += 1
    if nfin < MIN_VALID:
        return np.nan, np.nan
    smooth = y.copy()
    for i in range(122):
        if finite[i] == 0:
            continue
        vals = np.empty(5, np.float64)
        m = 0
        for k in range(-2, 3):
            j = i + k
            if 0 <= j < 122 and finite[j] == 1:
                vals[m] = y[j]
                m += 1
        if m == 0:
            continue
        for a in range(m):
            for b in range(a + 1, m):
                if vals[b] < vals[a]:
                    tmp = vals[a]
                    vals[a] = vals[b]
                    vals[b] = tmp
        smooth[i] = vals[m // 2]
    peak = -1
    peak_v = -1e9
    for i in range(122):
        if finite[i] == 1 and smooth[i] > peak_v:
            peak_v = smooth[i]
            peak = i
    if peak < 4 or peak > 117:
        return np.nan, np.nan
    # green-up: last minimum before the peak, else the start
    left = 0
    left_v = smooth[peak]
    for i in range(peak):
        if finite[i] == 1 and smooth[i] <= left_v:
            left_v = smooth[i]
            left = i
    right = 121
    right_v = smooth[peak]
    for i in range(peak, 122):
        if finite[i] == 1 and smooth[i] <= right_v:
            right_v = smooth[i]
            right = i
    ng = peak - left + 1
    ns = right - peak + 1
    if ng < 8 or ns < 8:
        return np.nan, np.nan
    tg = np.empty(ng, np.float64)
    yg = np.empty(ng, np.float64)
    for i in range(ng):
        tg[i] = t[left + i]
        yg[i] = smooth[left + i]
    ts = np.empty(ns, np.float64)
    ys = np.empty(ns, np.float64)
    for i in range(ns):
        ts[i] = t[peak + i]
        ys[i] = smooth[peak + i]
    t0g, kg, _ = _fit_logistic(tg, yg, True)
    t0s, ks, _ = _fit_logistic(ts, ys, False)
    sos = np.nan
    eos = np.nan
    if np.isfinite(t0g) and kg > 0:
        sos = t0g - DELTA / kg
        if sos < tg[0] or sos > tg[ng - 1]:
            sos = np.nan
    if np.isfinite(t0s) and ks > 0:
        eos = t0s + DELTA / ks
        if eos < ts[0] or eos > ts[ns - 1]:
            eos = np.nan
    if np.isfinite(sos) and np.isfinite(eos) and eos <= sos + 30.0:
        return np.nan, np.nan
    return sos, eos


@njit
def metrics_block(evi, sos, eos):
    n = evi.shape[0]
    for i in range(n):
        for year in range(N_YEARS):
            y = evi[i, year * STEPS:(year + 1) * STEPS]
            s, e = metrics_one_year(y)
            sos[i, year] = s
            eos[i, year] = e


def main() -> None:
    with Dataset(SRC) as src:
        n = len(src.dimensions["pixels"])
        log(f"pixels {n}")
        sos = np.full((n, N_YEARS), np.nan, np.float32)
        eos = np.full((n, N_YEARS), np.nan, np.float32)
        chunk = 4096
        for p0 in range(0, n, chunk):
            p1 = min(p0 + chunk, n)
            red = np.array(src.variables["data"][p0:p1, :, 2], np.float32)
            nir = np.array(src.variables["data"][p0:p1, :, 3], np.float32)
            series = evi2(red, nir)
            metrics_block(series, sos[p0:p1], eos[p0:p1])
            log(f"phenology {p1}/{n}")
        if DST.exists():
            DST.unlink()
        with Dataset(DST, "w", format="NETCDF4") as dst:
            dst.createDimension("pixels", n)
            dst.createDimension("year", N_YEARS)
            dst.createVariable("year", "i2", ("year",))[:] = np.arange(YEAR0, YEAR0 + N_YEARS, dtype=np.int16)
            vs = dst.createVariable("sos", "f4", ("pixels", "year"))
            ve = dst.createVariable("eos", "f4", ("pixels", "year"))
            vs.units = "day of year"
            ve.units = "day of year"
            vs.comment = "Green-up onset from the Zhang piecewise logistic curvature extremum."
            ve.comment = "Dormancy onset from the Zhang piecewise logistic curvature extremum."
            vs[:] = sos
            ve[:] = eos
            for name in ("lon", "lat", "row", "col", "lum_code", "dist_urban_m"):
                var = src.variables[name]
                copied = dst.createVariable(name, var.dtype, ("pixels",))
                copied[:] = var[:]
            dst.setncattr("method", "Xiaoyang Zhang hybrid piecewise logistic model on EVI2")
            dst.setncattr("evi2", "2.5*(NIR-Red)/(NIR+2.4*Red+1), reflectance = DN/10000")
            ok = np.isfinite(sos) & np.isfinite(eos)
            log(f"valid pixel-years {int(ok.sum())} / {ok.size}")
            dst.setncattr("n_valid_pixel_years", int(ok.sum()))
    log(f"wrote {DST}")


if __name__ == "__main__":
    main()
