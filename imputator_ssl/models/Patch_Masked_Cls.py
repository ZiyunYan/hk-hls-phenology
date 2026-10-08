import torch
import torch.nn as nn

from .Patch_Masked import PatchMaskedEncoder


class Model(nn.Module):
    """
    Patch_Masked 的监督分类版本：
    - 骨干与掩码自监督完全一致（PatchMaskedEncoder）
    - 不做 SSL 掩码，只用完整序列编码
    - 分类头：对最后一层 patch tokens 做 pooling，然后接 Linear -> num_class
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.encoder = PatchMaskedEncoder(configs)
        self.d_model = configs.d_model
        self.num_classes = configs.num_class

        # 分类头：全局 pooling + Linear
        self.classifier = nn.Linear(self.d_model, self.num_classes)
        self.geo_dropout_p = float(getattr(configs, "geo_dropout_p", 0.5))

    def _prepare_input(self, x_enc: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        准备输入：
        - x_clean: NaN -> 0
        - valid_mask: 同时考虑 NaN 和 padding（[B, T]）
        """
        B, T, _ = x_enc.shape
        device = x_enc.device

        # 原始有效性（按时间步）：任一通道为 NaN 就视为该时间步无效
        nan_valid = (~torch.isnan(x_enc).any(dim=-1)).float()  # [B, T]

        if padding_mask is not None:
            # padding_mask: [B, T]，1 = 有效，0 = padding
            valid_mask = nan_valid * padding_mask.float()
        else:
            valid_mask = nan_valid

        x_clean = torch.nan_to_num(x_enc, nan=0.0)
        return x_clean, valid_mask

    def forward(self, x_enc, padding_mask=None, time_mark=None, next_x_enc=None, mode='train', lon_lat=None, **kwargs):
        """
        与历史监督训练入口中的 batch 调用约定对齐：
        outputs = self.model(batch_x, padding_mask, None, None)

        Args:
            x_enc: [B, T, C]
            padding_mask: [B, T]，1 表示有效时间步，0 表示 padding
            time_mark: 目前不使用（保持接口兼容），可为 None
            lon_lat: 可选 [B, T, 2]，与 Patch_Masked / TED_modular 一致
        Returns:
            logits: [B, num_classes]
        """
        B, T, _ = x_enc.shape
        device = x_enc.device

        x_clean, valid_mask = self._prepare_input(x_enc, padding_mask)

        if time_mark is None:
            # 与 Patch_Masked 一致，time_mark 形状为 [B, T, 2]
            time_mark = torch.zeros(B, T, 2, device=device, dtype=x_enc.dtype)

        if lon_lat is None:
            lon_lat = kwargs.get("lon_lat", None)
        if lon_lat is not None:
            if lon_lat.shape[0] != B or lon_lat.shape[1] != T or lon_lat.shape[-1] != 2:
                raise ValueError(
                    f"lon_lat 期望 [B,T,2]，得到 {tuple(lon_lat.shape)}，与 x_enc [{B},{T},…] 不一致"
                )

        geo_keep = None
        if lon_lat is not None and getattr(self.encoder, "lon_lat_proj", None) is not None:
            if self.training and mode == "train" and self.geo_dropout_p > 0:
                geo_keep = (
                    torch.rand(B, 1, 1, device=device, dtype=torch.float32) >= self.geo_dropout_p
                ).to(dtype=x_enc.dtype)
            else:
                geo_keep = torch.ones(B, 1, 1, device=device, dtype=x_enc.dtype)

        # 不做 SSL 掩码，missing_mask 仅表示“原始有效 + 非 padding”的时间步
        missing_mask = valid_mask  # [B, T]

        # 编码：不传 mask_map（无 SSL 掩码）
        _, z_patch, z_cls = self.encoder(
            x_clean, missing_mask, time_mark, mask_map=None, lon_lat=lon_lat, geo_keep=geo_keep
        )  # [B, N_patches, D]

        # 与 ViT/BERT 及 TED 的「全局向量」对齐：使用 CLS 位而非 patch 均值
        logits = self.classifier(z_cls)  # [B, num_classes]
        return logits

    def encode(self, xEnc, timeMark=None, imputator=None, lon_lat=None, **kwargs):
        """
        与 TED 等模型对齐的 encode 接口：
        - 返回 cls_token (全局特征) 与 patch_tokens
        """
        B, T, _ = xEnc.shape
        device = xEnc.device

        # 原始有效性（不考虑 padding，这里默认下游会自己处理）
        missing_mask = (~torch.isnan(xEnc).any(dim=-1)).float()
        x_clean = torch.nan_to_num(xEnc, nan=0.0)

        if timeMark is None:
            timeMark = torch.zeros(B, T, 2, device=device, dtype=xEnc.dtype)

        if lon_lat is None:
            lon_lat = kwargs.get("lon_lat", None)
        if lon_lat is not None:
            if lon_lat.shape[0] != B or lon_lat.shape[1] != T or lon_lat.shape[-1] != 2:
                raise ValueError(
                    f"lon_lat 期望 [B,T,2]，得到 {tuple(lon_lat.shape)}，与 xEnc [{B},{T},…] 不一致"
                )

        _, z_patch, z_cls = self.encoder(
            x_clean, missing_mask, timeMark, mask_map=None, lon_lat=lon_lat, geo_keep=None
        )

        return {
            'cls_token': z_cls,
            'patch_tokens': z_patch,
        }
