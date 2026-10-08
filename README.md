# Hong Kong HLS phenology & imputation

End-to-end workflow for **Harmonized Landsat Sentinel-2 (HLS)** over Hong Kong (tile **49QHE**, 1830×1830 @ 30 m): 3-day compositing, stable-vegetation masking, Transformer gap-filling, and phenology / driver analysis.

## Data & models (Hugging Face)

| Asset | Hub repo |
|--------|-----------|
| Training pixels, full veg cube, metadata | [ZiyunPOLYU/hk-hls-phenology](https://huggingface.co/datasets/ZiyunPOLYU/hk-hls-phenology) |
| Best Imputator weights (`sl366`, topk5_s1) | [ZiyunPOLYU/hk-hls-imputator](https://huggingface.co/ZiyunPOLYU/hk-hls-imputator) |

After cloning this repo, download weights into:

`imputator_ssl/checkpoints/HK-Imputator-optical6-topk5-s1-sl366/checkpoint.pth`

```bash
hf download ZiyunPOLYU/hk-hls-imputator --local-dir imputator_ssl/checkpoints/HK-Imputator-optical6-topk5-s1-sl366
```

## Repository layout

```
hk_phenology/
├── composite_veg_cube.py      # Granule → 3-day nanmedian cube (full grid)
├── apply_veg_mask.py          # Drop agri; optional 2018∩2024 stable natural veg
├── impute_vegetation.py       # Grid imputation (366-day windows)
├── run_impute_fast.sh         # Multi-GPU sharded impute
├── phenology_hplm.py          # SOS/EOS (HPLM / Zhang-style on EVI2)
├── analyze_drivers.py         # Climate / terrain drivers
├── train_imputator_window_ablation.sh
└── imputator_ssl/             # Vendored training code (independent of TimeSeries_SSL_USA)
```

## Environment

- Python: `timesfm` conda env (`/home/ziyun218/.conda/envs/timesfm/bin/python`)
- PyTorch + CUDA, `netCDF4`, `rasterio`, `sktime` (for training dataloader)

## Typical pipeline

1. **Composite** (once): `composite_veg_cube.py` → `phenology/veg_cube.nc`
2. **Mask**: `apply_veg_mask.py` (no agriculture; stable 2018∩2024 natural vegetation)
3. **Impute**: `run_impute_fast.sh` or `impute_vegetation.py` → `veg_filled.nc`
4. **Phenology**: `phenology_hplm.py` → `phenology_sos_eos.nc`
5. **Drivers**: `analyze_drivers.py`

Cluster data root (not in Git): `/intelnvme03/ziyun218/hls_49QHE_hk/`

## Training the Imputator (HK-only)

```bash
bash train_imputator_window_ablation.sh
```

Runs `imputator_ssl/run.py` with `--data HLS` and `root_path` pointing at `imputator_hk/` NetCDF shards. Checkpoints are written under `imputator_ssl/checkpoints/`.

## Checkpoints (in GitHub)

| Model | Path | Inference |
|-------|------|-----------|
| **sl366** (production) | `imputator_ssl/checkpoints/HK-Imputator-optical6-topk5-s1-sl366/checkpoint.pth` | 3-year context → 1-year output |
| **sl122** | `.../HK-Imputator-optical6-topk5-s1-sl122/checkpoint.pth` | **一年一窗** (122 steps per forward) |
| **sl244** | `.../HK-Imputator-optical6-topk5-s1-sl244/checkpoint.pth` | **两年一窗** (244 steps per forward) |

Full rules: **[docs/IMPUTE_WINDOWS.md](docs/IMPUTE_WINDOWS.md)**.

## Citation & LULC

- HLS: https://hls.gsfc.nasa.gov/  
- Hong Kong LUM raster: Planning Department open data (2018, 2024 used for stable mask)

## License

Code: MIT (see `LICENSE`). Remote sensing data remain subject to NASA/USGS HLS terms and Hong Kong government open-data terms.
