---
license: mit
task_categories:
  - time-series-forecasting
language:
  - en
tags:
  - remote-sensing
  - landsat
  - sentinel-2
  - hls
  - hong-kong
  - phenology
  - imputation
size_categories:
  - 1G<n<10G
---

# Hong Kong HLS phenology bundle (49QHE)

## Contents

| Path | Description |
|------|-------------|
| `meta.json` | Grid size (1830×1830), UTM EPSG:32649 transform, tile id |
| `imputator_hk/train.nc` | ~37,484 pixels × 1342 time steps × 6 optical bands (int16 DN, 0=missing); `band_mean` / `band_std`; lon/lat |
| `imputator_hk/test.nc` | ~4,685 held-out pixels, same schema |
| `phenology/veg_cube.nc` | Full-grid cube `(time=1342, band=6, y=1830, x=1830)`; `vegetation` mask (stable natural veg, no agriculture); `lum_code` |

## Bands (6)

Blue, Green, Red, NIR (L30 B05 / S30 B8A), SWIR1, SWIR2 — reflectance × 10000 as `int16`, **0 = nodata**.

## Time

3-day compositing steps from HLS granules (~2015–2025); `time` variable is `%Y%j` strings.

## Vegetation mask

Natural vegetation codes 71–74; agriculture (61) removed; pixels kept only if natural in **both** 2018 and 2024 LUM and not urban in either year (~750k pixels).

## Related model

[ZiyunPOLYU/hk-hls-imputator](https://huggingface.co/ZiyunPOLYU/hk-hls-imputator) — Transformer gap-filler trained on `train.nc`.

## Usage

```bash
hf download ZiyunPOLYU/hk-hls-phenology --local-dir ./hk_hls_data
```

Point pipeline scripts at `HK_DATA_ROOT` or symlink to `/intelnvme03/...` layout:

```
hk_hls_data/meta.json
hk_hls_data/imputator_hk/train.nc
hk_hls_data/phenology/veg_cube.nc
```
