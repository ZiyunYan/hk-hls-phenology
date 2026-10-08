# sl122 — 1-year window imputator

| Item | Value |
|------|--------|
| `seq_len` / stride | **122** / **122** |
| Best val (train log) | **~0.01007** |
| Epoch | EarlyStopping best (60-epoch run, Oct 2026) |

## Inference rule

**一年一窗**：对 `year_index = 0…10`，每次只输入该年的 **122** 个 3 天步，模型输出同长度序列，**仅在原值为 0 处**写入。

详见 [`docs/IMPUTE_WINDOWS.md`](../../../docs/IMPUTE_WINDOWS.md).
