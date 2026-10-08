import torch
import torch.nn as nn
import numpy as np
import timesfm  # 确保安装 timesfm 库


def check_devices(model, input_tensor, name=""):
    """检查模型参数、缓冲区和输入张量的设备"""
    print(f"\n--- Checking devices for {name} ---")
    print(f"Input tensor device: {input_tensor.device}")
    for pname, param in model.named_parameters():
        if param.device != torch.device("cuda:0"):
            print(f"Parameter {pname} on {param.device}")
    for bname, buffer in model.named_buffers():
        if buffer.device != torch.device("cuda:0"):
            print(f"Buffer {bname} on {buffer.device}")


class Model(nn.Module):
    """
    TimesFM + Hook：默认对多波段做时间维聚合后走单变量 forecast；
    ``encode(..., per_band_concat=True)`` 时对每个波段分别 forecast，取各波段 last patch 向量再拼接（供 kNN 等多谱输入）。
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.dropout = nn.Dropout(p=0.2)

        if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            raise ImportError(
                "当前 `timesfm` 不是 2.5 发行线（例如从 PyPI 安装的 timesfm==1.3 仅有旧版 TimesFmTorch）。\n"
                "请卸载后从 GitHub 安装，例如：\n"
                '  pip uninstall -y timesfm\n'
                '  git clone https://github.com/google-research/timesfm.git && cd timesfm && pip install -e ".[torch]"\n'
                "首次 from_pretrained 需联网从 Hugging Face 下载 google/timesfm-2.5-200m-pytorch。"
            )

        self.imputation_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        self.imputation_model.compile(
            timesfm.ForecastConfig(
                max_context=732,
                max_horizon=32,
                normalize_inputs=False,
                use_continuous_quantile_head=False,
                force_flip_invariance=False,
                infer_is_positive=False,
                fix_quantile_crossing=False,
                per_core_batch_size=max(1, int(getattr(configs, "batch_size", 4)) // 4),
            )
        )

    def _forecast_patch_tokens_from_univariate_bt(
        self, series_bt: torch.Tensor, out_device: torch.device
    ) -> torch.Tensor:
        """
        series_bt: [B, T] 单变量时间序列（已在与 x_enc 相同的语义空间）
        返回 patch_tokens [B, N_patches, D_timesfm]
        """
        last_band = np.nan_to_num(
            series_bt.detach().cpu().float().numpy(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        batch_size = last_band.shape[0]
        inputs = [last_band[i] for i in range(batch_size)]

        seq_container = {}

        def hook_fn(module, input, output):
            # TimesFM_2p5_200M_torch_module.forward 返回:
            # ((input_embeddings, output_embeddings, output_ts, output_quantile_spread), new_decode_caches)
            seq_container["tokens"] = output[0][1].detach()

        handle = self.imputation_model.model.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                _point_forecast, _ = self.imputation_model.forecast(horizon=32, inputs=inputs)
        finally:
            handle.remove()

        patch_tokens = seq_container["tokens"].to(out_device)
        if int(patch_tokens.shape[0]) < int(batch_size):
            raise RuntimeError(
                f"TimesFM hook batch {patch_tokens.shape[0]} < 输入 batch {batch_size}，"
                f"请检查 compile 的 per_core_batch_size。"
            )
        return patch_tokens

    def _extract_patch_tokens(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        从 TimesFM 中提取 Transformer 输出的 patch-level embeddings。
        输入: x_enc [B, T, C]（已按你全局的 z-score 预处理）
        返回: patch_tokens [B, N_patches, D_timesfm]，其中每个 patch 覆盖 32 个时间步
        """
        # 默认：波段维 nanmean → 单变量 TimesFM
        combined = torch.nanmean(x_enc, dim=2)
        return self._forecast_patch_tokens_from_univariate_bt(combined, x_enc.device)

    def forward(self, x_enc, time_mark=None, valid_mask=None, next_x_enc=None, mode="train"):
        """
        输入: (batch, steps, bands)
        输出: (batch, D_timesfm) - 作为一个全局表示使用
        """
        patch_tokens = self._extract_patch_tokens(x_enc)  # [B, N_patches, D]
        last_tokens = patch_tokens[:, -1, :]

        if mode == "train" or mode == "fine-tune":
            return None, None, last_tokens, None
        elif mode == "pred":
            return last_tokens, last_tokens, last_tokens, last_tokens

    def encode(self, xEnc, timeMark=None, imputator=None, per_band_concat=False, **kwargs):
        """
        与 TED 接口兼容的 encode（供 KNN / 分类脚本读 ``cls_token``）：
        - 输入: xEnc [B, T, C]
        - per_band_concat: True 时每个波段单独走 TimesFM 单变量分支，将各波段 last-patch 向量在最后一维拼接（维度 C*D）。
        - 输出:
            'cls_token': [B, D_timesfm] 或 [B, C*D_timesfm]（per_band_concat）
            'patch_tokens': [B, T, D] 或 [B, T, C*D]（各波段 patch 序列沿特征维拼接后的对齐展开）
        """
        patch_len = 32
        T_in = int(xEnc.shape[1])

        if per_band_concat:
            last_vecs = []
            patch_blocks = []
            for c in range(int(xEnc.shape[2])):
                band = xEnc[:, :, c]
                patch_tokens_p = self._forecast_patch_tokens_from_univariate_bt(band, xEnc.device)
                last_vecs.append(patch_tokens_p[:, -1, :])
                time_tokens = patch_tokens_p.repeat_interleave(patch_len, dim=1)
                patch_blocks.append(time_tokens[:, -T_in:, :])
            cls_token = torch.cat(last_vecs, dim=-1)
            patch_tokens = torch.cat(patch_blocks, dim=-1)
            return {
                "cls_token": cls_token,
                "patch_tokens": patch_tokens,
            }

        patch_tokens_p = self._extract_patch_tokens(xEnc)
        time_tokens = patch_tokens_p.repeat_interleave(patch_len, dim=1)
        patch_tokens = time_tokens[:, -T_in:, :]
        cls_token = patch_tokens.mean(dim=1)
        return {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
        }

    def encode_sequence(self, xEnc, timeMark=None, imputator=None, **kwargs):
        """
        显式的序列编码接口（与 TED 的 encode 语义对齐）：
        - 返回同 encode，一般用于变化检测等 patch-level 分析。
        """
        return self.encode(xEnc, timeMark=timeMark, imputator=imputator, **kwargs)
