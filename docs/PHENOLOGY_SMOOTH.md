# Phenology smoothing (30 m)

Imputation does **not** remove observation noise. Before Zhang HPLM on EVI2, apply a **simple smoother**:

| Method | Notes |
|--------|--------|
| **`sg9`** (default) | Savitzky–Golay, window 9 (~27 days), poly 2 — good balance |
| `sg7` / `sg11` | Less / more smoothing |
| `median7` | Rolling median; weak on spikes |
| `hampel7_median7` | Despike then median |

## Quick compare (~2 min first run, tiled I/O)

```bash
bash run_try_smooth.sh
# or
python try_smooth_methods.py -n 120 --seed 20261008
```

Uses `veg_filled.nc` if present, else `veg_cube.nc` (unfilled → fewer valid SOS/EOS; still compares smoothers).

## Run phenology

```bash
# Subset test (fast)
python phenology_from_cube.py --smooth sg9 --max-pixels 3000

# Full vegetation grid (slow)
python phenology_from_cube.py --smooth sg9
```

Point-level nc (`pixels` dim): `python phenology_hplm.py --smooth sg9`
