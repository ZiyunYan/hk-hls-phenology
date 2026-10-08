---
license: mit
library_name: pytorch
tags:
  - time-series
  - imputation
  - transformer
  - hls
  - hong-kong
---

# HK HLS Imputator (Transformer, sl366)

Best checkpoint for Hong Kong 6-band optical HLS imputation.

## Weights

- File: `checkpoint.pth` (~19 MB)
- Architecture: 6-layer Transformer encoder, `d_model=256`, `n_heads=8`, `d_ff=1024`
- `seq_len=366`, training stride 122 (3-day steps)

## Training summary

- Data: `ZiyunPOLYU/hk-hls-phenology` → `imputator_hk/train.nc`
- Loss: MSE + dy2 smoothness, top-5 trim per sequence, `smooth_beta=1`
- Selected: epoch **51**, validation loss minimum

## Load (inference)

Use code in [GitHub `hk-hls-phenology`](https://github.com/ZiyunYan/hk-hls-phenology) → `impute_vegetation.py` or:

```python
from pathlib import Path
import torch
from models.Transformer import Model  # with imputator_ssl on PYTHONPATH
# ... see impute_vegetation.py configs() and CKPT path
```

## Download

```bash
hf download ZiyunPOLYU/hk-hls-imputator checkpoint.pth --local-dir ./HK-Imputator-optical6-topk5-s1-sl366
```
