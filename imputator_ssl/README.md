# Hong Kong Imputator (vendored from TimeSeries_SSL_USA)

Self-contained copy of training/inference code for HK HLS imputation. **Edits here do not change** `/home/ziyun218/pyprojects/TimeSeries_SSL_USA`.

## Layout

- `run.py` — training CLI (`--data HLS`, `--model Transformer`)
- `exp/`, `models/`, `layers/`, `utils/`, `data_provider/` — copied snapshot (2026-10-08)
- `checkpoints/` — HK weights for this project (not shared with USA repo)

## Data & weights (Hub)

- Dataset: https://huggingface.co/datasets/ZiyunPOLYU/hk-hls-phenology  
- Best checkpoint: https://huggingface.co/ZiyunPOLYU/hk-hls-imputator  

Local cluster path (optional): `/intelnvme03/ziyun218/hls_49QHE_hk/imputator_hk/{train,test}.nc`

## Train

```bash
bash /home/ziyun218/pyprojects/hk_phenology/train_imputator_window_ablation.sh
```

Runs from this directory; new checkpoints land under `imputator_ssl/checkpoints/`.

## Re-sync code from upstream (optional)

```bash
cd /home/ziyun218/pyprojects
SRC=TimeSeries_SSL_USA
DST=hk_phenology/imputator_ssl
rsync -a "$SRC/run.py" "$DST/"
rsync -a "$SRC/exp/" "$DST/exp/"
rsync -a "$SRC/models/" "$DST/models/"
rsync -a "$SRC/layers/" "$DST/layers/"
rsync -a "$SRC/utils/" "$DST/utils/"
rsync -a "$SRC/data_provider/" "$DST/data_provider/"
cp "$SRC/analysis/analyze_synthetic_trajectories.py" "$DST/analysis/"
```

Review diffs before overwriting HK-specific patches.

## Inference

`../impute_vegetation.py` and `../preview_recon.py` import `Model` from this tree.
