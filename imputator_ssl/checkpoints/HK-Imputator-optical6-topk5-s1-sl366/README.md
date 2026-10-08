# HK Imputator — production checkpoint

| Field | Value |
|--------|--------|
| Model | Transformer Imputator, 6 optical bands |
| `seq_len` | 366 (3-day steps, ~3-year context) |
| Recipe | `topk=5`, `smooth_beta=1`, `smooth_mode=dy2`, `scaling_rule=none` |
| Training data | `imputator_hk/train.nc` (~37k pixels, tile 49QHE) |
| Selection | Lowest validation loss; **epoch 51** |
| Inference | **3-year context → 1-year output** (11 passes); see `docs/IMPUTE_WINDOWS.md` |
| Hub | [ZiyunPOLYU/hk-hls-imputator](https://huggingface.co/ZiyunPOLYU/hk-hls-imputator) |

Place `checkpoint.pth` in this directory (not committed to Git; download from Hugging Face).
