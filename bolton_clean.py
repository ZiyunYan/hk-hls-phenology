#!/usr/bin/env python3
"""Pre-interpolation cleaning of the 3-day composite, before any gap filling.

The bright-anomaly test follows Bolton et al. (2020), RSE 240:111685, section
3.3.1. The spike test keeps their 0.1 threshold and the requirement that the
point sit beyond both neighboring clear observations, and changes three
choices for Hong Kong:

- the index is Huete EVI, not EVI2
- spikes are removed in both directions
- the gap between the previous and next clear observations may be up to 90 days

Locked rules:

1. Bright anomalies, adapted from MAJA (Hagolle et al., 2017). An observation
   is removed when its blue reflectance jumps relative to both neighbors,

       blue(D) - blue(Dnb) > 0.03 * (1 + |D - Dnb| / 30)

   unless the red jump is larger still, which Bolton treats as vegetation
   change rather than cloud:

       red(D) - red(Dnb) > 1.5 * (blue(D) - blue(Dnb))

2. EVI spikes. With the previous and next clear observations less than 90 days
   apart, the point is removed when EVI is more than 0.1 above the time-linear
   interpolation and above both neighbors, or more than 0.1 below the
   interpolation and below both neighbors. All six bands of that date are set
   to missing. The flag is computed from the spectra; the spectra are what get
   deleted.

Snow filling, the cubic spline, and the topographic illumination correction
are not applied. Hong Kong has no separate snow-QA map, and Fmask snow was
already set to missing in the composite.
"""
from __future__ import annotations

import numpy as np

BLUE, RED, NIR = 0, 2, 3
SPIKE_SPAN_DAYS = 90.0
SPIKE_DELTA = 0.1


def reflectance(dn: np.ndarray) -> np.ndarray:
    return np.clip(dn.astype(np.float32) / 10000.0, 0.0, 1.0)


def evi2_from_dn(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """EVI2, kept for the earlier gap-count scripts. The locked clean uses EVI."""
    red_r = reflectance(red)
    nir_r = reflectance(nir)
    den = nir_r + 2.4 * red_r + 1.0
    out = np.full(red_r.shape, np.nan, np.float32)
    ok = np.abs(den) > 1e-6
    out[ok] = 2.5 * (nir_r[ok] - red_r[ok]) / den[ok]
    return out


def evi_from_dn(blue: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Huete EVI. Same reflectance scaling as phenology_hplm.evi."""
    blue_r = reflectance(blue)
    red_r = reflectance(red)
    nir_r = reflectance(nir)
    den = nir_r + 6.0 * red_r - 7.5 * blue_r + 1.0
    out = np.full(red_r.shape, np.nan, np.float32)
    ok = np.abs(den) > 1e-6
    out[ok] = 2.5 * (nir_r[ok] - red_r[ok]) / den[ok]
    return out


def _neighbor_index(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strictly previous and next valid index. valid is (pixels, time)."""
    n, steps = valid.shape
    ar = np.arange(steps, dtype=np.int32)
    prev_filled = np.where(valid, ar, np.int32(-1))
    prev_filled = np.maximum.accumulate(prev_filled, axis=1)
    prev = np.full((n, steps), np.int32(-1))
    prev[:, 1:] = prev_filled[:, :-1]
    next_filled = np.where(valid, ar, np.int32(steps))
    next_filled = np.minimum.accumulate(next_filled[:, ::-1], axis=1)[:, ::-1]
    nxt = np.full((n, steps), np.int32(steps))
    nxt[:, :-1] = next_filled[:, 1:]
    return prev, nxt


def _gather(values: np.ndarray, index: np.ndarray, empty: float) -> np.ndarray:
    flat = values.reshape(-1)
    offset = (np.arange(values.shape[0], dtype=np.int64) * values.shape[1])[:, None]
    take = index.astype(np.int64) + offset
    ok = (index >= 0) & (index < values.shape[1])
    out = np.full(index.shape, empty, np.float32)
    out[ok] = flat[take[ok]]
    return out


def bolton_outlier_mask(cube: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cube (pixels, time, 6) DN. day (time,) ordinal days.

    Returns boolean masks (pixels, time) for bright anomalies and EVI spikes.
    A time step is eligible only when blue, red, and NIR are all non-zero.
    The spike mask includes both directions.
    """
    blue = reflectance(cube[:, :, BLUE])
    red = reflectance(cube[:, :, RED])
    valid = (cube[:, :, BLUE] > 0) & (cube[:, :, RED] > 0) & (cube[:, :, NIR] > 0)
    evi = evi_from_dn(cube[:, :, BLUE], cube[:, :, RED], cube[:, :, NIR])
    evi = np.where(valid, evi, np.nan)
    prev, nxt = _neighbor_index(valid)
    has_both = (prev >= 0) & (nxt < cube.shape[1]) & valid
    day = np.asarray(day, np.float32)
    day_grid = np.broadcast_to(day, valid.shape)
    day_prev = _gather(day_grid, prev, np.nan)
    day_next = _gather(day_grid, nxt, np.nan)
    blue_prev = _gather(blue, prev, np.nan)
    blue_next = _gather(blue, nxt, np.nan)
    red_prev = _gather(red, prev, np.nan)
    red_next = _gather(red, nxt, np.nan)

    def bright(current_b, current_r, other_b, other_r, gap):
        dblue = current_b - other_b
        dred = current_r - other_r
        threshold = 0.03 * (1.0 + gap / 30.0)
        landcover = dred > 1.5 * dblue
        return (dblue > threshold) & ~landcover

    gap_prev = np.abs(day[None, :] - day_prev)
    gap_next = np.abs(day_next - day[None, :])
    bright_mask = (
        has_both
        & bright(blue, red, blue_prev, red_prev, gap_prev)
        & bright(blue, red, blue_next, red_next, gap_next)
    )

    evi_prev = _gather(np.where(valid, evi, 0.0), prev, np.nan)
    evi_next = _gather(np.where(valid, evi, 0.0), nxt, np.nan)
    span = day_next - day_prev
    weight = (day[None, :] - day_prev) / np.where(span > 0, span, np.nan)
    fitted = evi_prev * (1.0 - weight) + evi_next * weight
    base = has_both & np.isfinite(fitted) & (span > 0) & (span < SPIKE_SPAN_DAYS)
    down = base & ((fitted - evi) > SPIKE_DELTA) & (evi_prev > evi) & (evi_next > evi)
    up = base & ((evi - fitted) > SPIKE_DELTA) & (evi > evi_prev) & (evi > evi_next)
    return bright_mask, down | up


def apply_bolton_clean(cube: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Return a copy of cube with outlier observations set to 0."""
    bright, spike = bolton_outlier_mask(cube, day)
    out = np.array(cube, copy=True)
    drop = bright | spike
    out[drop] = 0
    stats = {
        "observed": int(((cube[:, :, BLUE] > 0) & (cube[:, :, RED] > 0) & (cube[:, :, NIR] > 0)).sum()),
        "bright": int(bright.sum()),
        "spike": int((spike & ~bright).sum()),
        "removed": int(drop.sum()),
    }
    return out, stats


def _self_check() -> None:
    day = np.arange(5, dtype=np.float32) * 10.0
    cube = np.zeros((3, 5, 6), np.int16)

    def put(pixel, step, blue, red, nir):
        cube[pixel, step, BLUE] = int(blue * 10000)
        cube[pixel, step, RED] = int(red * 10000)
        cube[pixel, step, NIR] = int(nir * 10000)

    for step, blue in enumerate((0.04, 0.04, 0.20, 0.04, 0.04)):
        put(0, step, blue, 0.05, 0.30)
    for step, red in enumerate((0.05, 0.05, 0.20, 0.05, 0.05)):
        put(1, step, 0.06 if step == 2 else 0.04, red, 0.30)
    for step, nir in enumerate((0.40, 0.40, 0.12, 0.40, 0.40)):
        put(2, step, 0.04, 0.05, nir)
    bright, spike = bolton_outlier_mask(cube, day)
    # High blue is a bright anomaly and also an upward EVI spike.
    assert bright[0, 2] and spike[0, 2]
    assert not bright[1].any()
    assert spike[2, 2] and not bright[2].any()
    # Same dip with neighbors 100 days apart stays. 90 days is the cap.
    day_wide = np.array([0, 10, 40, 110, 120], np.float32)
    bright_w, spike_w = bolton_outlier_mask(cube, day_wide)
    assert not spike_w[2, 2]
    # A 60-day gap is inside the locked 90-day window.
    day_60 = np.array([0, 10, 40, 70, 80], np.float32)
    _, spike_60 = bolton_outlier_mask(cube, day_60)
    assert spike_60[2, 2]
    high = np.zeros((1, 5, 6), np.int16)
    for step, nir in enumerate((0.25, 0.25, 0.55, 0.25, 0.25)):
        high[0, step, BLUE] = 400
        high[0, step, RED] = 500
        high[0, step, NIR] = int(nir * 10000)
    _, spike_up = bolton_outlier_mask(high, day)
    assert spike_up[0, 2]
    print("bolton_clean self-check ok")


if __name__ == "__main__":
    _self_check()
