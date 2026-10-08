"""
掩码自监督模型 (Masked Self-Supervised Learning)

与 TED_modular 所用 Backbone 刻意对齐的部分：
  patchify 几何、Linear(patch)、time_mark / missing_mask 嵌入、傅里叶 lon_lat + geo_keep、
  register(storage) tokens、可学习 cls_token、PositionalEncoding 仅加在 patch 序列上、
  同构 AttentionBlock（含 drop_path）、最终 LayerNorm。

训练目标（与 TED 的本质差异，论文中应写明）：
  「patch 置零 → 解码重建」的 MSE；无 teacher EMA、无 DINO/iBOT/Koleo；
  掩码在 patch 值域置零后再过 Linear 嵌入（与 Backbone iBOT 路径的 mask_token 替换不同）。

--- PatchTST 范式对标 (Nie et al., “A Time Series is Worth 64 Words”, ICLR 2023, §3.2) ---

论文中掩码自监督要点：
  (1) 子序列级 patch 作为 Transformer token；(2) 预训练用**非重叠** patch（stride 等于 patch 长），
      使可见 patch 不泄漏被掩 patch 的原始值；(3) 按索引**均匀随机**选取约固定比例（如 40%）的 patch，
      用**零**替换；(4) 仅对被掩 patch 做 MSE 重建；(5) **通道独立（channel independence）**：
      每个变量单独过同一套骨干（共享权重）。

本实现与 PatchTST 的对齐关系：
  - 已对齐 (1)(2)(4)：patch token + 当 stride==patch_len 时为非重叠；损失仅在 SSL 掩码位置。
  - (3) 默认对齐：``patchtst_style_masking=1`` 时使用 ``random_patch_masking_patchtst_uniform``，
    默认 ``patchtst_mask_ratio=0.4``（PatchTST §3.2 表示学习设置）。设 ``patchtst_style_masking=0`` 则回退为
    DINOv3 式块掩码 + ``mask_rate_v1/v2`` 区间。
  - **未**实现 (5)：当前为**通道混合** patch 嵌入 (patch_len * C → d_model)，与 PatchTST 的逐通道独立前向不同；
    若要与论文完全同构，需另建 CI 分支（每通道重复 B 维或循环 C 次骨干）。

encode() 的 cls_token 与 Backbone 一致取序列首位，便于与 TED / NTP 的 KNN probe 可比。
"""

import torch
import torch.nn as nn
import math

from layers.Self_layers import AttentionBlock
from layers.Embedding import PositionalEncoding
from modules.backbone import lon_lat_fourier_features
from utils.tools import patchify, random_patch_masking_dinov3_style, random_patch_masking_patchtst_uniform
from utils.losses import mse_loss


class PatchMaskedEncoder(nn.Module):
    """
    与 modules.backbone.Backbone 的「patch 支路」对齐：CLS + storage + patch tokens、
    同构 AttentionBlock（含 drop_path）；无 DINO/iBOT head、无 mask_token 嵌入替换（见类文档）。
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
        self.num_patches = math.ceil((self.seq_len - self.patch_len + self.stride) / self.stride)
        self.n_storage_tokens = getattr(configs, 'n_storage_tokens', 2)
        self.missing_mask_embed_dropout = float(
            getattr(configs, "missing_mask_embed_dropout", 0.0)
        )

        # Embedding（与 TED / Backbone 一致）
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
        # 说明：本模型采用 “patchify 后把被 mask 的 patch 置 0，再做 embedding”
        # 因此不再使用 mask_token 在嵌入后替换（与 Backbone iBOT 路径的 mask_token 不同）。

        # 与 Backbone 一致：可学习 CLS + register(storage)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, self.n_storage_tokens, self.d_model))
            nn.init.normal_(self.storage_tokens, std=0.02)

        # Encoder（与 Backbone 相同 AttentionBlock 配置，含 stochastic depth）
        _attn_ckpt = not getattr(configs, 'no_attn_checkpoint', False)
        self.encoder = nn.ModuleList([
            AttentionBlock(
                self.d_model,
                self.n_heads,
                configs.d_ff // self.d_model,
                self.dropout,
                drop_path=self.drop_path,
                use_checkpoint=_attn_ckpt,
            )
            for _ in range(self.e_layers)
        ])
        self.norm = nn.LayerNorm(self.d_model)

        # 线性层：将 d_model 映射回 patch_len * c_out
        self.patch_decode_head = nn.Linear(self.d_model, self.patch_len * self.c_out)

    def _lon_lat_patch_embed(
        self,
        lon_lat: torch.Tensor,
        B: int,
        N: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """与 Backbone._lon_lat_patch_embed 一致：每样本取时间首步经纬度 → 傅里叶特征 → d_model，广播到各 patch。"""
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

    def forward(self, xEnc, missing_mask, time_mark, mask_map=None, lon_lat=None, geo_keep=None, output_attentions=False):
        """
        Args:
            xEnc: [B, T, C]
            missing_mask: [B, T]
            time_mark: [B, T, 2]
            mask_map: [B, N] bool, True 表示该 patch 被 mask
            lon_lat: 可选 [B, T, 2]，WGS84 度；与 Backbone 一致用 [:,0,:] 作为像元位置
            geo_keep: 可选 [B,1,1]，训练时 CFG 丢弃地理嵌入（与 TED_modular 一致）
        Returns:
            rec_patches: [B, N, patch_len * c_out] 重建的 patch
            z_patch: [B, N, D] 仅 patch 位的 encoder 输出
            z_cls: [B, D] 与 Backbone 首位 CLS 对齐
        """
        # 1. Patchify
        x_patches = patchify(xEnc, self.patch_len, self.stride)  # [B, N, patch_len*C]
        B, N, _ = x_patches.shape

        # 2. Patch-level masking（直接在 patch 空间置 0，再做 embedding）
        if mask_map is not None:
            # mask_map: [B, N]，True 表示该 patch 被 mask
            x_patches = x_patches * (~mask_map).unsqueeze(-1).type_as(x_patches)

        # 3. Embedding
        x_embed = self.embedding(x_patches)  # [B, N, D]
        D = x_embed.shape[-1]

        if time_mark is not None:
            time_mark_patches = patchify(time_mark, self.patch_len, self.stride)
            time_mark_embed = self.time_mark_embedding(time_mark_patches)
        else:
            time_mark_embed = torch.zeros(B, N, D, device=xEnc.device, dtype=x_embed.dtype)

        missing_mask_patches = patchify(missing_mask.unsqueeze(-1).float(), self.patch_len, self.stride)
        missing_mask_embed = self.missing_mask_embedding(missing_mask_patches)
        missing_mask_embed = self._maybe_drop_missing_mask_embed(missing_mask_embed)

        ll_emb = self._lon_lat_patch_embed(lon_lat, B, N, x_embed.dtype, xEnc.device)
        if geo_keep is not None:
            ll_emb = ll_emb * geo_keep.to(dtype=ll_emb.dtype)

        x_input = x_embed + time_mark_embed + missing_mask_embed + ll_emb

        # 4. 位置编码
        pos_embed = self.position_encoding(x_input)
        if pos_embed.shape[1] > N:
            pos_embed = pos_embed[:, :N, :]
        elif pos_embed.shape[1] < N:
            pad_len = N - pos_embed.shape[1]
            pos_embed = torch.cat([
                pos_embed,
                torch.zeros(B, pad_len, D, device=pos_embed.device, dtype=pos_embed.dtype)
            ], dim=1)
        x_input = x_input + pos_embed

        # 5. 与 Backbone 一致：[CLS]、[storage…]、patch tokens（位置编码仅作用于 patch 段）
        cls_tok = self.cls_token.expand(B, -1, -1)
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens.expand(B, -1, -1)
            x_input = torch.cat([cls_tok, storage_tokens, x_input], dim=1)
        else:
            x_input = torch.cat([cls_tok, x_input], dim=1)

        # 6. Encoder（非因果）
        all_attns = []
        for layer in self.encoder:
            x_input, attn = layer(
                x_input, is_causal=False, return_attn=bool(output_attentions)
            )
            if output_attentions:
                all_attns.append(attn)
        x_input = self.norm(x_input)

        z_cls = x_input[:, 0, :]
        if self.n_storage_tokens > 0:
            z_patch = x_input[:, 1 + self.n_storage_tokens :, :]
        else:
            z_patch = x_input[:, 1:, :]

        rec_patches = self.patch_decode_head(z_patch)
        if output_attentions:
            return rec_patches, z_patch, z_cls, all_attns, x_input
        return rec_patches, z_patch, z_cls


class Model(nn.Module):
    """
    掩码自监督：patch 级随机置零 + MSE 重建被掩码 patch（见模块级 docstring 与 PatchTST 对标说明）。
    """

    def __init__(self, configs):
        super().__init__()
        self.encoder = PatchMaskedEncoder(configs)
        self.configs = configs
        # 固定长度训练：在 Exp_SSL（exp_ssl.py）中用于决定是否启用 mixed_batch（梯度累积+变长裁剪）
        # 这里显式提供该属性，避免走默认 'fast' 或 'mixed_batch'
        self.curriculum_strategy = getattr(configs, 'curriculum_strategy', 'none')
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.c_out = configs.c_out
        self.num_patches = math.ceil((configs.seq_len - configs.patch_len + configs.stride) / configs.stride)
        self.mask_rate_v1 = getattr(configs, 'mask_rate_v1', 0.3)
        self.mask_rate_v2 = getattr(configs, 'mask_rate_v2', 0.6)
        self.block_mask_ratio = getattr(configs, 'block_mask_ratio', 0.8)
        self.geo_dropout_p = float(getattr(configs, "geo_dropout_p", 0.5))
        self.patchtst_style_masking = bool(int(getattr(configs, "patchtst_style_masking", 1)))
        self.patchtst_mask_ratio = float(getattr(configs, "patchtst_mask_ratio", 0.4))

    def _prepare_input(self, xEnc, valid_mask=None):
        """
        准备输入：
        - 返回填充 NaN 后的原始序列 x_clean
        - 返回基于 NaN 的原始有效性掩码 valid_mask（只用于 loss，不直接作为 SSL mask）
        """
        if valid_mask is None:
            # [B, T]：任一通道为 NaN 就视为该时间步缺失
            valid_mask = (1 - torch.isnan(xEnc).any(dim=-1).int())
        x_clean = torch.nan_to_num(xEnc, nan=0.0)
        return x_clean, valid_mask.float()

    def forward(self, xEnc, time_mark=None, valid_mask=None, next_x_enc=None, mode='train',
                mask_rate_v1=None, mask_rate_v2=None, lon_lat=None, **kwargs):
        """
        Args:
            xEnc: [B, T, C]
            time_mark: [B, T, 2]
            valid_mask: [B, T] 有效点 mask
            lon_lat: 可选 [B, T, 2]，与 TED_modular / dataloader 一致
            mode: 'train' | 'pred' | 'test'
        """
        B = xEnc.shape[0]
        device = xEnc.device
        x_clean, valid_mask = self._prepare_input(xEnc, valid_mask)
        if time_mark is None:
            time_mark = torch.zeros(B, xEnc.shape[1], 2, device=device, dtype=xEnc.dtype)

        if lon_lat is not None:
            if lon_lat.shape[0] != B or lon_lat.shape[1] != xEnc.shape[1] or lon_lat.shape[-1] != 2:
                raise ValueError(
                    f"lon_lat 期望 [B,T,2]，得到 {tuple(lon_lat.shape)}，与 xEnc [{B},{xEnc.shape[1]},…] 不一致"
                )

        geo_keep = None
        if lon_lat is not None and getattr(self.encoder, "lon_lat_proj", None) is not None:
            if self.training and mode == "train" and self.geo_dropout_p > 0:
                geo_keep = (
                    torch.rand(B, 1, 1, device=device, dtype=torch.float32) >= self.geo_dropout_p
                ).to(dtype=xEnc.dtype)
            else:
                geo_keep = torch.ones(B, 1, 1, device=device, dtype=xEnc.dtype)

        mv1 = mask_rate_v1 if mask_rate_v1 is not None else self.mask_rate_v1
        mv2 = mask_rate_v2 if mask_rate_v2 is not None else self.mask_rate_v2

        # 当前序列长度对应的 patch 数（支持 mixed-batch 下的变长 T）
        T_cur = xEnc.shape[1]
        num_patches_cur = math.ceil((T_cur - self.patch_len + self.stride) / self.stride)

        if self.patchtst_style_masking:
            # PatchTST §3.2：固定比例 + 均匀随机 patch 索引（非块掩码）
            mask_map, _, _ = random_patch_masking_patchtst_uniform(
                B=B,
                num_patches=num_patches_cur,
                mask_ratio=self.patchtst_mask_ratio,
                device=device,
            )
        else:
            # DINOv3 风格：比例区间 + 块掩码混合
            mask_map, _, _ = random_patch_masking_dinov3_style(
                B=B,
                mask_ratio_tuple=(mv1, mv2),
                mask_sample_probability=1.0,
                num_patches=num_patches_cur,
                device=device,
                block_ratio=self.block_mask_ratio,
            )

        # 将 patch-level SSL 掩码映射回序列，用作 encoder 的 missing_mask（模型需要知道哪些位置是“需要预测”的）
        ssl_mask_patches = mask_map.float().unsqueeze(-1).expand(-1, -1, self.patch_len)  # [B, N, patch_len]
        from utils.tools import unpatchify
        ssl_missing_mask_seq = unpatchify(ssl_mask_patches, xEnc.shape[1], self.patch_len, 1).squeeze(-1)  # [B, T]

        rec_patches, z_patch, z_cls = self.encoder(
            x_clean, ssl_missing_mask_seq, time_mark, mask_map, lon_lat=lon_lat, geo_keep=geo_keep
        )

        # 目标：原始 patch
        target_patches = patchify(x_clean, self.patch_len, self.stride)

        if mode == 'train':
            # 在原始序列空间计算 loss，并叠加两层 mask：
            # 1) patch mask：只对被 mask 的 patch 计算
            # 2) valid mask：只对原始序列中非 NaN 的位置计算（按通道）
            from utils.tools import unpatchify

            pred_seq = unpatchify(rec_patches, xEnc.shape[1], self.patch_len, self.c_out)  # [B, T, C]
            target_seq = x_clean[:, :pred_seq.shape[1], :self.c_out]  # [B, T, C]

            # patch mask -> sequence mask
            mask_patches = mask_map.float().unsqueeze(-1).expand(-1, -1, self.patch_len * self.c_out)
            patch_mask_seq = unpatchify(mask_patches, xEnc.shape[1], self.patch_len, self.c_out)  # [B, T, C]

            # valid mask（原始有效性，按通道）：NaN=无效
            valid_mask_seq = (~torch.isnan(xEnc[:, :pred_seq.shape[1], :self.c_out])).float()

            loss_mask = patch_mask_seq * valid_mask_seq
            loss = mse_loss(pred_seq, target_seq, loss_mask)
            return {
                'ssl_loss': loss,
                'rec_patches': rec_patches,
                'target_patches': target_patches,
                'mask_map': mask_map,
                'log_vars': {'mse_recon': loss.item()},
            }
        else:
            # pred/test: 返回重建序列（unpatchify）；cls_token 与 Backbone / TED encode 一致为首位 token
            from utils.tools import unpatchify
            rec_seq = unpatchify(rec_patches, xEnc.shape[1], self.patch_len, self.c_out)
            return {
                'rec_seq': rec_seq,
                'z_patch': z_patch,
                'cls_token': z_cls,
            }

    def encode(
        self,
        xEnc,
        timeMark=None,
        imputator=None,
        lon_lat=None,
        pool_mode: str = "patch_avg",
        **kwargs,
    ):
        """
        编码接口，供 KNN / linear probe 等下游使用。
        这里的 missing_mask 表示原始缺失（NaN），而非 SSL 掩码。

        pool_mode:
            - ``patch_avg``（默认）：对所有 patch token 均值池化；MSM 无 CLS 监督项时更合理。
            - ``cls``：使用首位 CLS token（历史结果与此一致）。
        返回 ``cls_token`` 为下游探针向量（命名沿用 TED 接口）。
        """
        _ = imputator
        pool_mode = kwargs.pop("pool_mode", pool_mode)
        if pool_mode not in ("patch_avg", "cls"):
            raise ValueError(f"pool_mode must be 'patch_avg' or 'cls', got {pool_mode!r}")

        B = xEnc.shape[0]
        device = xEnc.device
        missing_mask = (1 - torch.isnan(xEnc).any(dim=-1).int()).float()
        x_clean = torch.nan_to_num(xEnc, nan=0.0)
        if timeMark is None:
            timeMark = torch.zeros(B, xEnc.shape[1], 2, device=device, dtype=xEnc.dtype)
        elif timeMark.dim() == 2:
            timeMark = timeMark.unsqueeze(-1).expand(-1, -1, 2)
        if lon_lat is None:
            lon_lat = kwargs.get("lon_lat", None)
        if lon_lat is not None:
            if lon_lat.shape[0] != B or lon_lat.shape[1] != xEnc.shape[1] or lon_lat.shape[-1] != 2:
                raise ValueError(
                    f"lon_lat 期望 [B,T,2]，得到 {tuple(lon_lat.shape)}，与 xEnc [{B},{xEnc.shape[1]},…] 不一致"
                )
        _, z_patch, z_cls = self.encoder(
            x_clean, missing_mask, timeMark, mask_map=None, lon_lat=lon_lat, geo_keep=None
        )
        if pool_mode == "patch_avg":
            embedding = z_patch.mean(dim=1)
        else:
            embedding = z_cls
        return {
            "cls_token": embedding,
            "patch_tokens": z_patch,
            "cls_raw": z_cls,
            "pool_mode": pool_mode,
        }
