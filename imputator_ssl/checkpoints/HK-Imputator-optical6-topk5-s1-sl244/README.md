# sl244 — 2-year window imputator

| Item | Value |
|------|--------|
| `seq_len` / stride | **244** / **244** |
| Best val (99 训练日志) | **~0.016007** (epoch 59) |
| Trained on | `10.21.17.99`, `Time_Series_SSL` |

## Inference rule

**两年一窗**：对 `block = 0…4`，每次输入 **244** 步（两年），填补该段内缺测；步 **1220–1341**（最后一年）用 **sl122 单年窗** 或单独 122 步推理。

详见 [`docs/IMPUTE_WINDOWS.md`](../../../docs/IMPUTE_WINDOWS.md).
