"""
Next-token prediction（patch 级一步预测），与 TED_modular / Patch_Masked 的输入编码对齐：

- patchify（patch_len / stride 与 TED 一致）+ Linear(patch)
- time_mark 嵌入、missing_mask 嵌入（含训练期 missing_mask_embed_dropout）
- 傅里叶 lon/lat 地理嵌入 + geo_keep（与 Backbone / Patch_Masked 一致）
- PositionalEncoding 仅加在 patch 序列上
- 因果 AttentionBlock（含 drop_path）

目标：同一 732 时间窗内，每 patch 预测「下一 patch」（stride=3 即时间上错位 3 步），
pred[:, i] 对齐 target[:, i+1]，MSE；不再使用 dataloader 的 delay / next_x_enc 外推窗。
"""

import math

import torch
import torch.nn as nn

from layers.Self_layers import AttentionBlock
from layers.Embedding import PositionalEncoding
from modules.backbone import lon_lat_fourier_features
from utils.losses import mse_loss
from utils.tools import patchify, unpatchify


class PatchNTPTEDEncoder(nn.Module):
    """
    NTP Encoder：与 Patch_Masked 的 patch 支路编码对齐（无 CLS / storage）。
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.n_heads = configs.n_heads
        self.dropout = float(getattr(configs, "dropout", 0.0))
        self.drop_path = float(
            getattr(configs, "drop_path", getattr(configs, "drop_depth", 0.0))
        )
        self.e_layers = configs.e_layers
        self.num_patches = math.ceil(
            (self.seq_len - self.patch_len + self.stride) / self.stride
        )
        self.missing_mask_embed_dropout = float(
            getattr(configs, "missing_mask_embed_dropout", 0.0)
        )

        patch_dim = self.patch_len * self.enc_in
        self.embedding = nn.Linear(patch_dim, self.d_model)
        self.time_mark_embedding = nn.Linear(self.patch_len * 2, self.d_model)
        self.missing_mask_embedding = nn.Linear(self.patch_len * 1, self.d_model)
        self.use_lon_lat_embed = bool(getattr(configs, "use_lon_lat_embed", True))
        self.lon_lat_n_fourier_freqs = int(getattr(configs, "lon_lat_n_fourier_freqs", 4))
        _fourier_dim = 4 * self.lon_lat_n_fourier_freqs
        if self.use_lon_lat_embed:
            self.lon_lat_proj = nn.Linear(_fourier_dim, self.d_model)
        else:
            self.lon_lat_proj = None
        self.position_encoding = PositionalEncoding(self.d_model, self.num_patches)

        _attn_ckpt = not getattr(configs, "no_attn_checkpoint", False)
        self.encoder = nn.ModuleList(
            [
                AttentionBlock(
                    self.d_model,
                    self.n_heads,
                    configs.d_ff // self.d_model,
                    self.dropout,
                    drop_path=self.drop_path,
                    use_checkpoint=_attn_ckpt,
                )
                for _ in range(self.e_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.patch_decode_head = nn.Linear(self.d_model, self.patch_len * self.c_out)

    def _lon_lat_patch_embed(
        self,
        lon_lat: torch.Tensor,
        B: int,
        N: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self.lon_lat_proj is None or lon_lat is None:
            return torch.zeros(B, N, self.d_model, device=device, dtype=dtype)
        rep = torch.nan_to_num(lon_lat[:, 0, :].float(), nan=0.0)
        feat = lon_lat_fourier_features(rep, self.lon_lat_n_fourier_freqs).to(dtype=dtype)
        ll = self.lon_lat_proj(feat)
        return ll.unsqueeze(1).expand(B, N, -1)

    def _maybe_drop_missing_mask_embed(self, missing_mask_embed: torch.Tensor) -> torch.Tensor:
        if (
            not self.training
            or self.missing_mask_embed_dropout <= 0.0
            or missing_mask_embed is None
        ):
            return missing_mask_embed
        B = int(missing_mask_embed.shape[0])
        device = missing_mask_embed.device
        keep = (
            torch.rand(B, 1, 1, device=device, dtype=torch.float32)
            >= self.missing_mask_embed_dropout
        ).to(dtype=missing_mask_embed.dtype)
        return missing_mask_embed * keep

    def forward(
        self,
        xEnc,
        missing_mask,
        time_mark,
        lon_lat=None,
        geo_keep=None,
        output_attentions=False,
    ):
        x_patches = patchify(xEnc, self.patch_len, self.stride)
        x_embed = self.embedding(x_patches)
        B, N, D = x_embed.shape

        if time_mark is not None:
            time_mark_patches = patchify(time_mark, self.patch_len, self.stride)
            time_mark_embed = self.time_mark_embedding(time_mark_patches)
        else:
            time_mark_embed = torch.zeros(B, N, D, device=xEnc.device, dtype=x_embed.dtype)

        missing_mask_patches = patchify(
            missing_mask.unsqueeze(-1).float(), self.patch_len, self.stride
        )
        missing_mask_embed = self.missing_mask_embedding(missing_mask_patches)
        missing_mask_embed = self._maybe_drop_missing_mask_embed(missing_mask_embed)

        ll_emb = self._lon_lat_patch_embed(lon_lat, B, N, x_embed.dtype, xEnc.device)
        if geo_keep is not None:
            ll_emb = ll_emb * geo_keep.to(dtype=ll_emb.dtype)

        x_input = x_embed + time_mark_embed + missing_mask_embed + ll_emb

        pos_embed = self.position_encoding(x_input)
        if pos_embed.shape[1] > N:
            pos_embed = pos_embed[:, :N, :]
        elif pos_embed.shape[1] < N:
            pad_len = N - pos_embed.shape[1]
            pos_embed = torch.cat(
                [
                    pos_embed,
                    torch.zeros(
                        B, pad_len, D, device=pos_embed.device, dtype=pos_embed.dtype
                    ),
                ],
                dim=1,
            )
        x_input = x_input + pos_embed

        all_attns = []
        for layer in self.encoder:
            x_input, attn = layer(x_input, is_causal=True, return_attn=output_attentions)
            if output_attentions:
                all_attns.append(attn)
        z = self.norm(x_input)
        rec_patches = self.patch_decode_head(z)
        if output_attentions:
            return rec_patches, z, all_attns
        return rec_patches, z, None


class Model(nn.Module):
    """
    Patch 级 next-step：位置 i 的解码输出预测第 i+1 个 patch（时间上前移 stride 步）。
    """

    def __init__(self, configs):
        super().__init__()
        self.encoder = PatchNTPTEDEncoder(configs)
        self.configs = configs
        self.curriculum_strategy = getattr(configs, "curriculum_strategy", "none")
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.c_out = configs.c_out
        self.seq_len = configs.seq_len
        self.num_patches = math.ceil(
            (configs.seq_len - configs.patch_len + configs.stride) / configs.stride
        )
        self.geo_dropout_p = float(getattr(configs, "geo_dropout_p", 0.5))

    def _prepare_input(self, xEnc, valid_mask=None):
        if valid_mask is None:
            valid_mask = (1 - torch.isnan(xEnc).any(dim=-1).int())
        x_clean = torch.nan_to_num(xEnc, nan=0.0)
        missing_mask = valid_mask.float()
        return x_clean, missing_mask

    def forward(self, xEnc, time_mark=None, valid_mask=None, next_x_enc=None, mode="train", **kwargs):
        """
        next_x_enc: 保留参数以兼容训练循环；本模型训练时不再使用（目标为窗内 patch_{i+1}）。
        """
        _ = next_x_enc  # 不使用 delay/next 窗；与训练循环传参兼容
        B = xEnc.shape[0]
        device = xEnc.device
        x_clean, missing_mask = self._prepare_input(xEnc, valid_mask)
        if time_mark is None:
            time_mark = torch.zeros(B, xEnc.shape[1], 2, device=device, dtype=xEnc.dtype)

        lon_lat = kwargs.get("lon_lat", None)
        if lon_lat is not None:
            if lon_lat.shape[0] != B or lon_lat.shape[1] != xEnc.shape[1] or lon_lat.shape[-1] != 2:
                raise ValueError(
                    f"lon_lat 期望 [B,T,2]，得到 {tuple(lon_lat.shape)}，与 x_enc [{B},{xEnc.shape[1]},…] 不一致"
                )

        geo_keep = None
        if lon_lat is not None and getattr(self.encoder, "lon_lat_proj", None) is not None:
            if self.training and mode == "train" and self.geo_dropout_p > 0:
                geo_keep = (
                    torch.rand(B, 1, 1, device=device, dtype=torch.float32) >= self.geo_dropout_p
                ).to(dtype=xEnc.dtype)
            else:
                geo_keep = torch.ones(B, 1, 1, device=device, dtype=xEnc.dtype)

        output_attentions = (mode in ("pred", "test")) and not self.training
        result = self.encoder(
            x_clean,
            missing_mask,
            time_mark,
            lon_lat=lon_lat,
            geo_keep=geo_keep,
            output_attentions=output_attentions,
        )
        if len(result) == 3:
            rec_patches, z, all_attns = result
        else:
            rec_patches, z = result[:2]
            all_attns = None

        if mode == "train":
            target_patches = patchify(x_clean, self.patch_len, self.stride)
            pred_shifted = rec_patches[:, :-1, :]
            target_shifted = target_patches[:, 1:, :]
            valid_mask_seq = (~torch.isnan(xEnc[:, : x_clean.shape[1], : self.c_out])).float()
            valid_mask_patches = patchify(
                valid_mask_seq, self.patch_len, self.stride
            )
            valid_mask_shifted = valid_mask_patches[:, 1:, :]
            loss = mse_loss(pred_shifted, target_shifted, valid_mask_shifted)
            return {
                "ssl_loss": loss,
                "rec_patches": rec_patches,
                "target_patches": target_patches,
                "log_vars": {
                    "mse_ntp": loss.item(),
                    "ntp_target": "next_patch_in_window",
                },
            }

        target_patches = patchify(x_clean, self.patch_len, self.stride)
        rec_seq = unpatchify(rec_patches, xEnc.shape[1], self.patch_len, self.c_out)
        last_token = z[:, -1, :]
        return {
            "rec_seq": rec_seq,
            "z_patch": z,
            "last_token": last_token,
            "all_attns": all_attns,
        }

    def encode(self, xEnc, timeMark=None, imputator=None, lon_lat=None, **kwargs):
        """
        无 cls：用 last_token（最后一个 patch 特征）作为 cls_token，供 KNN probe 等对齐调用习惯。
        """
        _ = imputator
        B = xEnc.shape[0]
        device = xEnc.device
        missing_mask = (1 - torch.isnan(xEnc).any(dim=-1).int()).float()
        x_clean = torch.nan_to_num(xEnc, nan=0.0)
        if timeMark is None:
            timeMark = torch.zeros(B, xEnc.shape[1], 2, device=device, dtype=xEnc.dtype)
        elif timeMark.dim() == 2:
            timeMark = timeMark.unsqueeze(-1).expand(-1, -1, 2)
        if lon_lat is not None and (
            lon_lat.shape[0] != B
            or lon_lat.shape[1] != xEnc.shape[1]
            or lon_lat.shape[-1] != 2
        ):
            raise ValueError(
                f"lon_lat 期望 [B,T,2]，得到 {tuple(lon_lat.shape)}，与 x_enc [{B},{xEnc.shape[1]},…] 不一致"
            )
        _, z, _ = self.encoder(
            x_clean,
            missing_mask,
            timeMark,
            lon_lat=lon_lat,
            geo_keep=None,
            output_attentions=False,
        )
        last_token = z[:, -1, :]
        return {"cls_token": last_token, "patch_tokens": z}
