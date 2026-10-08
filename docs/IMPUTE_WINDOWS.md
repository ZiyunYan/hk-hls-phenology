# Hong Kong grid imputation — window length & checkpoint

All models share the same recipe: **6 optical bands**, `topk=5`, `smooth_beta=1`, `dy2`, HK `band_mean` / `band_std` from `imputator_hk/train.nc`.  
Time axis: **3-day steps**, **122 steps ≈ 1 calendar year**, full series **1342 steps** (~11 years).

Only **gap steps** (`DN == 0`) are overwritten; observed DN are kept.

---

## Checkpoints in this repo (`imputator_ssl/checkpoints/`)

| Folder | `seq_len` | Training stride | Val loss (approx.) | Inference schedule |
|--------|-----------|-----------------|--------------------|--------------------|
| `HK-Imputator-optical6-topk5-s1-sl366` | **366** | 122 | ~0.011 | **3-year context**, output **1 year** (middle year for interior windows) — default `impute_vegetation.py` |
| `HK-Imputator-optical6-topk5-s1-sl122` | **122** | 122 | ~0.010 | **1 year in, 1 year out** — year by year |
| `HK-Imputator-optical6-topk5-s1-sl244` | **244** | 244 | ~0.016 | **2 years in, 2 years out** — two-year blocks |

---

## sl122 — infer **year by year** (一年一窗)

- **Window length** `seq_len = 122` (= one year of 3-day composites).
- **Loop** `year_index = 0 … 10`:
  - Slice input: `t0 = year_index * 122`, `t1 = t0 + 122`
  - Run model once on `data[:, t0:t1, :]` (z-scored, `mode=pred`).
  - Write predictions only where `veg_cube[:, t0:t1, :] == 0`.
- **11 forward passes** per pixel per full timeline.
- **No** multi-year context: each call sees exactly **one** year.

```text
Year:  0      1      2     …     10
       [122]  [122]  [122]       [122]
        ↑      ↑      ↑           ↑
      1×model per block
```

---

## sl244 — infer **two years at a time** (两年一窗)

- **Window length** `seq_len = 244` (= two years).
- **Loop** `block_index = 0 … 4`:
  - Slice: `t0 = block_index * 244`, `t1 = t0 + 244` (covers steps **0–1219**).
  - One model call per block; fill gaps only inside that 244-step span.
- **5 forward passes** cover **5 × 244 = 1220** steps.
- **Remaining 122 steps** (`1220 … 1341`, the **11th year**): use **sl122 year-by-year** on that slice only, or run one sl122 window on `[1220:1342)`.

```text
Block:  0–1y   2–3y   4–5y   6–7y   8–9y   | 10y only
        [244]  [244]  [244]  [244]  [244]  | [122] ← sl122
```

---

## sl366 — default (three-year context, one-year output)

Used for production `veg_filled.nc` and Hugging Face `hk-hls-imputator`.  
See `impute_vegetation.py` → `year_window()`:

- First / last year: 366-step window aligned to series start/end; take first or last 122 steps.
- Interior years: 366-step window centered on the target year; take **middle** 122 steps.

**11 forward passes** per pixel; stronger temporal context, best val among ablations for full-series fill.

---

## Code entry

Default grid fill: `impute_vegetation.py` (currently wired for **sl366** only).  
For **sl122 / sl244**, use the same `Model` + `mode=pred` but change the temporal loop as in the tables above (`seq_len` and stride must match the checkpoint).
