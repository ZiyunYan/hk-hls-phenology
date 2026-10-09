#!/usr/bin/env python3
"""Pre-interpolation cleaning from Bolton et al. (2020), RSE 240:111685.

Section 3.3.1, applied to the 3-day composite before any gap filling.
Two tests, both required to agree with the preceding and following clear
observations:

1. Bright anomalies, adapted from MAJA (Hagolle et al., 2017). An observation
   is removed when its blue reflectance jumps relative to both neighbors,

       blue(D) - blue(Dnb) > 0.03 * (1 + |D - Dnb| / 30)

   unless the red jump is larger still, which Bolton treats as vegetation
   change rather than cloud:

       red(D) - red(Dnb) > 1.5 * (blue(D) - blue(Dnb))

2. Negative EVI2 spikes from missed cloud shadow. With the gap between the
   previous and next clear observations under 45 days, the point is removed
   when it lies more than 0.1 below the time-linear interpolation and below
   both neighbors.

Snow filling with the 5th percentile snow-free EVI2, the cubic spline, and
the topographic illumination correction are not applied here. Hong Kong has
no separate snow-QA map, and Fmask snow was already set to missing in the
composite, so the NDWI>0.5 and 5 km snow-QA screen has nothing to attach to.
"""
from __future__ import annotations

import numpy as np

BLUE, RED, NIR = 0, 2, 3


def reflectance(dn: np.ndarray) -> np.ndarray:
    return np.clip(dn.astype(np.float32) / 10000.0, 0.0, 1.0)


def evi2_from_dn(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    red_r = reflectance(red)
    nir_r = reflectance(nir)
    den = nir_r + 2.4 * red_r + 1.0
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

    Returns boolean masks (pixels, time) for bright anomalies and EVI2 spikes.
    A time step is eligible only when blue, red, and NIR are all non-zero.
    """
    blue = reflectance(cube[:, :, BLUE])
    red = reflectance(cube[:, :, RED])
    valid = (cube[:, :, BLUE] > 0) & (cube[:, :, RED] > 0) & (cube[:, :, NIR] > 0)
    evi = evi2_from_dn(cube[:, :, RED], cube[:, :, NIR])
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
    spike = (
        has_both
        & (span < 45.0)
        & np.isfinite(fitted)
        & ((fitted - evi) > 0.1)
        & (evi_prev > evi)
        & (evi_next > evi)
    )
    return bright_mask, spike


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
    assert bright[0, 2] and not spike[0, 2]
    assert not bright[1].any()
    assert spike[2, 2] and not bright[2].any()
    # Same dip, but the neighbors are 60 days apart, so the spike rule does not fire.
    day_wide = np.array([0, 10, 40, 70, 80], np.float32)
    bright_w, spike_w = bolton_outlier_mask(cube, day_wide)
    assert not spike_w[2, 2]
    print("bolton_clean self-check ok")


if __name__ == "__main__":
    _self_check()
