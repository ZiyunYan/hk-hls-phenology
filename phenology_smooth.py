"""Fast 1D smoothers for annual EVI2 curves (122 steps, 3-day sampling)."""
from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover
    median_filter = None
    savgol_filter = None

SMOOTH_METHODS = (
    "median5",  # legacy (in HPLM)
    "median7",
    "median9",
    "sg7",
    "sg9",
    "sg11",
    "hampel7_median7",
)


def _nan_median_filter(y: np.ndarray, size: int) -> np.ndarray:
    if median_filter is None:
        raise RuntimeError("scipy required for median smooth")
    out = y.astype(np.float64, copy=True)
    bad = ~np.isfinite(out)
    if bad.all():
        return out.astype(np.float32)
    fill = np.nanmedian(out)
    out[bad] = fill
    sm = median_filter(out, size=size, mode="nearest")
    sm[bad] = np.nan
    return sm.astype(np.float32)


def hampel(y: np.ndarray, window: int = 7, n_sigma: float = 3.0) -> np.ndarray:
    """Replace spikes with local median (Hampel)."""
    y = y.astype(np.float64, copy=True)
    n = y.size
    half = window // 2
    for i in range(n):
        if not np.isfinite(y[i]):
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        w = y[lo:hi]
        w = w[np.isfinite(w)]
        if w.size < 3:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad < 1e-6:
            continue
        if abs(y[i] - med) > n_sigma * 1.4826 * mad:
            y[i] = med
    return y.astype(np.float32)


def smooth_annual(y: np.ndarray, method: str) -> np.ndarray:
    """Smooth one year EVI2, shape (122,)."""
    method = method.lower()
    if method == "none":
        return y.astype(np.float32, copy=True)
    if method == "median5":
        return _median5_numba_style(y)
    if method.startswith("median"):
        k = int(method.replace("median", ""))
        return _nan_median_filter(y, k)
    if method.startswith("sg"):
        if savgol_filter is None:
            raise RuntimeError("scipy required for Savitzky–Golay")
        w = int(method.replace("sg", ""))
        if w % 2 == 0 or w < 5:
            raise ValueError(f"SG window must be odd >= 5, got {w}")
        out = y.astype(np.float64, copy=True)
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = np.nanmedian(out[np.isfinite(out)])
        sm = savgol_filter(out, window_length=w, polyorder=2, mode="interp")
        sm[bad] = np.nan
        return sm.astype(np.float32)
    if method == "hampel7_median7":
        return _nan_median_filter(hampel(y, 7), 7)
    raise ValueError(f"unknown smooth method {method}; choose from {SMOOTH_METHODS}")


def _median5_numba_style(y: np.ndarray) -> np.ndarray:
    """Match original phenology_hplm 5-point rolling median."""
    n = y.size
    smooth = y.astype(np.float32, copy=True)
    finite = np.isfinite(y)
    for i in range(n):
        if not finite[i]:
            continue
        vals = []
        for k in range(-2, 3):
            j = i + k
            if 0 <= j < n and finite[j]:
                vals.append(float(y[j]))
        if vals:
            smooth[i] = np.median(vals)
    return smooth


def smooth_block(years: np.ndarray, method: str) -> np.ndarray:
    """years shape (..., 122) — smooth along last axis."""
    flat = years.reshape(-1, years.shape[-1])
    out = np.empty_like(flat, dtype=np.float32)
    for i in range(flat.shape[0]):
        out[i] = smooth_annual(flat[i], method)
    return out.reshape(years.shape)
