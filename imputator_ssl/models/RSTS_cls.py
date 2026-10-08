import torch
import torch.nn as nn
import einops
import numpy as np
import math
from utils.losses import mse_loss, smooth_loss, nll_loss, huber_loss, nll_loss_student_t, calculate_nll_t_per_timestep, \
    calculate_nll_per_timestep, dispersive_loss
from utils.tools import apply_mask, process_attention
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from layers.Self_layers import make_2tuple, Swish, RotaryPositionEmbedding1D, PatchEmbed, ResidualVectorQuantizer, InterSeasonBlock, IntraSeasonBlock, UpsampleBlock, DownsampleBlock, IntraSeason_AttnBlock, InterSeason_AttnBlock


class Encoder(nn.Module):
    def __init__(self, steps, bands, steps_per_season, dim=64, num_heads=4, num_groups=2, num_register_tokens=4, mlp_ratio=2,
                 order=["intra", "inter"], dropout=0.1):
        super().__init__()
        self.steps = steps
        self.bands = bands
        self.steps_per_season = steps_per_season
        self.dim = dim
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        
        assert dim % num_heads == 0
        
        # Patch embedding
        self.patch_embed = PatchEmbed(seq_len=steps, kernel=steps_per_season, stride=steps_per_season,
            in_chans=bands+2, embed_dim=dim)
    
        self.season = self.patch_embed.num_patches
        # print('season', self.season)
        self.padding = self.patch_embed.padding
        
        # Register tokens
        self.intra_register_token = nn.Parameter(torch.randn(1, num_register_tokens, dim))
        nn.init.normal_(self.intra_register_token, std=1e-6)
        self.inter_register_token = nn.Parameter(torch.randn(1, num_register_tokens, dim))
        nn.init.normal_(self.inter_register_token, std=1e-6)
        
        # Attention blocks
        self.order = order
        self.intra_attn = IntraSeason_AttnBlock(
            dim=dim, num_heads=num_heads, num_register_tokens=num_register_tokens,
            mlp_ratio=mlp_ratio, rope_frequency=50, num_attn_blocks=self.num_groups//2, dropout=dropout
        )
        
        self.inter_attn = InterSeason_AttnBlock(
            dim=dim, num_heads=num_heads, num_register_tokens=num_register_tokens,
            mlp_ratio=mlp_ratio, rope_frequency=100, num_attn_blocks=self.num_groups, dropout=dropout
        )
    
    def forward(self, x, mask=None, time_mark=None):
        B = x.size(0)
        
        # 拼接输入 + Patch embedding
        x = torch.cat([x, time_mark], dim=-1)
        x = self.patch_embed(x)  # (B, season, steps_per_season, dim)
        # print('patch_embed x.shape', x.shape)
        # ✅ 修复后的 mask 处理
        intra_mask, inter_mask = self._get_masks(B, mask)

        
        # 按顺序执行
        for attn_type in self.order:
            if attn_type == "intra":
                x = self._add_register_tokens(x, self.intra_register_token, dim=2)
                x, _ = self.intra_attn(x, mask=intra_mask)
                x = self._remove_register_tokens(x, dim=2, keep_cls=True)
                x = x[:, :, 0, :].unsqueeze(2)
                # print('intra_attn output:x.shape', x.shape)
                
            elif attn_type == "inter":
                # x_inter = x.permute(0, 2, 1, 3)  # [B, steps, season, dim]
                # print('inter_attn input before add register tokens:x_inter.shape', x_inter.shape)
                x_inter = self._add_register_tokens(x, self.inter_register_token, dim=1)
                x_inter, _ = self.inter_attn(x_inter, mask=None)
                x_inter = self._remove_register_tokens(x_inter, dim=1)
                # x = x_inter.permute(0, 2, 1, 3)  # [B, season, steps, dim]
        
        return x, intra_mask, inter_mask
    
    def _get_masks(self, B, mask=None):
        """✅ 修复：自动添加 register token 部分"""
        num_register_tokens = self.num_register_tokens
        
        if mask is not None:
            # Padding + reshape
            mask = torch.nn.functional.pad(mask, (0, self.padding), mode='constant', value=0)
            mask = mask.view(B, self.season, self.steps_per_season)
            
            # Intra mask
            intra_mask_base = (1 - mask.view(B * self.season, self.steps_per_season)).to(mask.device)
            intra_register_mask = torch.zeros(B * self.season, num_register_tokens, 
                                            dtype=intra_mask_base.dtype, device=intra_mask_base.device)
            intra_mask = torch.cat([intra_register_mask, intra_mask_base], dim=1)
            
            # Inter mask
            inter_mask_base = (1 - mask.view(B * self.steps_per_season, self.season)).to(mask.device)
            inter_register_mask = torch.zeros(B * self.steps_per_season, num_register_tokens, 
                                            dtype=inter_mask_base.dtype, device=inter_mask_base.device)
            inter_mask = torch.cat([inter_register_mask, inter_mask_base], dim=1)
            
        else:
            intra_mask = None
            inter_mask = None
        
        return intra_mask, inter_mask

    def _add_register_tokens(self, x, register_token, dim):
        """在指定维度前添加 register tokens"""
        B, N, S, D = x.shape

        if dim == 1:  # INTER
            # register_token: [B, 5, 1, dim] 
            # x:              [B, season, steps, dim]
            register_token = register_token.unsqueeze(2).expand(B, -1, S, -1)  # [B, 5, 1, dim]
            x = torch.cat([register_token, x], dim=1)     # [B, 5+season, steps, dim]
            
        elif dim == 2:  # INTRA
            # register_token: [B, 1, 5, dim] 
            # x:              [B, steps, season, dim]
            register_token = register_token.unsqueeze(1).expand(B, N, -1, -1)  
            x = torch.cat([register_token, x], dim=2)     # [B, season, steps+5, dim]
        
        return x

    def _remove_register_tokens(self, x, dim, keep_cls=False):
        """
        从指定维度去掉 register tokens，保留 1 个 CLS token
        
        Args:
            x: 输入张量
            dim: 移除维度 (1=intra, 2=inter)
            keep_cls: 是否保留 CLS token (默认True)
        """
        num_register = self.num_register_tokens  # 4
        num_keep = 1 if keep_cls else 0
        
        if dim == 1:  # intra
            start_idx = num_register - num_keep  # 3 (保留第4个作为CLS)
            x = x[:, start_idx:, ...]
        elif dim == 2:  # inter
            start_idx = num_register - num_keep  # 3
            x = x[:, :, start_idx:, ...]
        
        return x


# class ImputationHead(nn.Module):
#     def __init__(self, dim, total_steps, kernel, stride, bands, num_layers=2):
#         super().__init__()
#         self.dim = dim
#         self.total_steps = total_steps
#         self.kernel = kernel
#         self.stride = stride
#         self.bands = bands
        
#         # Calculate number of patches (same as PatchEmbedding)
#         self.num_patches = math.ceil((total_steps - kernel + stride) / stride)
#         if self.num_patches < 1:
#             raise ValueError(f"Kernel size {kernel} is larger than total_steps {total_steps}")
        
#         # Calculate steps per patch
#         steps_per_patch = []
#         for i in range(self.num_patches):
#             start = i * stride
#             end = min(start + kernel, total_steps)
#             steps = end - start
#             steps_per_patch.append(steps)
        
#         self.steps_per_patch = steps_per_patch
#         assert sum(self.steps_per_patch) == total_steps, f"Steps per patch sum {sum(self.steps_per_patch)} does not match total_steps {total_steps}"
#         # print(f"Steps per patch: {self.steps_per_patch}")
        
#         # MLP to map from dim to max_steps * bands
#         max_steps = max(self.steps_per_patch)  # Use maximum steps for MLP output
#         layers = []
#         for i in range(num_layers):
#             in_dim = dim if i == 0 else dim * 2
#             layers.extend([nn.LayerNorm(in_dim), nn.Linear(in_dim, dim * 2), nn.GELU()])
#         layers.append(nn.Linear(dim * 2, max_steps * bands))
        
#         self.head = nn.Sequential(*layers)
    
#     def forward(self, patch_features):
#         B, P, T, D = patch_features.shape
#         assert P == self.num_patches, f"Expected {self.num_patches} patches, got {P}"
#         assert T == 1, f"Expected 1 token per patch, got {T}"
#         assert D == self.dim, f"Expected dim {self.dim}, got {D}"
        
#         imputation_out = []
#         for p in range(self.num_patches):
#             # Extract token for patch p: [B, 1, dim] -> [B, dim]
#             patch_feat = patch_features[:, p, 0, :]  # [B, dim]
#             # Pass through MLP: [B, dim] -> [B, max_steps * bands]
#             patch_pred = self.head(patch_feat)
#             # Reshape to [B, max_steps, bands]
#             patch_pred = patch_pred.view(B, -1, self.bands)
#             # Slice to actual steps for this patch: [B, steps_per_patch[p], bands]
#             patch_pred = patch_pred[:, :self.steps_per_patch[p], :]
#             imputation_out.append(patch_pred)
        
#         # Concatenate along time dimension: [B, total_steps, bands]
#         output = torch.cat(imputation_out, dim=1)
#         assert output.shape[1] == self.total_steps, f"Output length {output.shape[1]} does not match total_steps {self.total_steps}"
#         # print(f"Output shape: {output.shape}")
#         return output

class ImputationHead(nn.Module):
    def __init__(self, dim, bands, steps, num_heads=4, num_layers=1):
        super().__init__()
        self.dim = dim
        self.bands = bands
        self.steps = steps
        
        # MLP for embedding original sequence to dim
        self.q_embed = nn.Sequential(
            nn.Linear(bands, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
        
        # Cross attention with batch_first=True
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # MLP for reconstruction
        layers = []
        for i in range(num_layers):
            in_dim = dim if i == 0 else dim * 2
            layers.extend([nn.LayerNorm(in_dim), nn.Linear(in_dim, dim * 2), nn.GELU()])
        layers.append(nn.Linear(dim * 2, bands))
        self.reconstructor = nn.Sequential(*layers)
    
    def forward(self, original, season_features):
        # original: (batch, steps, bands)
        # season_features: (batch, seasons, dim)
        batch, steps, bands = original.shape
        season_features = season_features.squeeze(2)
        
        # Embed original sequence
        q = self.q_embed(original)  # (batch, steps, dim)

        # Cross attention (no permute needed due to batch_first=True)
        cross_out, attn_weights = self.cross_attn(
            query=q,  # (batch, steps, dim)
            key=season_features,  # (batch, seasons, dim)
            value=season_features  # (batch, seasons, dim)
        )  # cross_out: (batch, steps, dim), attn_weights: (batch, steps, seasons)
        
        # Reconstruction
        output = self.reconstructor(cross_out)  # (batch, steps, bands)
        
        assert output.shape == (batch, self.steps, self.bands), \
            f"Output shape {output.shape} does not match expected {(batch, self.steps, self.bands)}"
        
        return output


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.encoder = Encoder(steps=configs.seq_len, bands=configs.enc_in, steps_per_season=14, 
        dim=configs.d_model, num_heads=configs.n_heads, num_groups=6, mlp_ratio=configs.mlp_ratio, num_register_tokens=configs.num_register_tokens, dropout=configs.dropout)
        # self.imputation_head = ImputationHead(dim=configs.d_model, total_steps=configs.seq_len, kernel=14, stride=14, bands=configs.enc_in, num_layers=2)
        self.imputation_head = ImputationHead(dim=configs.d_model, bands=configs.enc_in, steps=configs.seq_len, num_heads=4, num_layers=1)

        self.configs = configs
        self.dropout = nn.Dropout(p=0.2)
        print('mask ratio:', configs.mask_rate)

    def _prepare_forecast_label(self, next_x):
        """Prepare input tensor with padding and masking."""

        next_valid_mask = (1 - torch.isnan(next_x).int()).to('cuda')   

        # Apply mask
        next_batch, _, _, _ = apply_mask(
            next_x, next_valid_mask, 0, 'cuda'
        )

        return next_batch, next_valid_mask

    def _prepare_input(self, x_enc, mask_ratio, valid_mask=None, mode='train'):
        """Prepare input tensor with padding and masking."""
        # batch_size, seq_len, bands = x_enc.shape
        if valid_mask is None:
            valid_mask = (1 - torch.isnan(x_enc).int()).to('cuda')

        # Apply mask
        batch_x, batch_x_masked, missing_mask, indicating_mask = apply_mask(
            x_enc, valid_mask, mask_ratio, 'cuda', mode
        )

        return batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask


    def _train(self, x_enc, time_mark, valid_mask=None, next_x_enc=None, mode='imputation'):
        """
        VAE 训练的前向传播，支持 imputation、forecast 和 anomaly 模式。

        参数：
            x_enc: 输入编码，形状为 (batch_size, seq_len, feature_dim)
            time_mark: 时间标记
            valid_mask: 有效掩码，默认为 None
            next_x_enc: 预测模式下的目标数据，默认为 None
            mode: 训练模式，'imputation', 'forecast' 或 'anomaly'

        返回：
            pred: 模型预测输出
            loss: 总损失（包括重建损失、vq_loss 和 Dispersive Loss）
            anomaly_output: 异常模式输出（用于后续处理）
            imputation_output: 填补模式输出（用于后续处理）
        """

        # 准备输入
        batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
            x_enc, self.configs.mask_rate, valid_mask
        )

        time_mark = self.dropout(time_mark)
        # ViT 前向传播，获取输出、vq_loss 和 inter_features
        output, inter_features, attn_outputs = self.encoder(batch_x_masked, mask=missing_mask[:, :, 0].float(), time_mark=time_mark)
        imputation_output = self.imputation_head(batch_x_masked, output)
        loss = imputation_loss(imputation_output, batch_x, indicating_mask, alpha=0.8, beta=0.2)
        return (
            imputation_output,
            loss,
            imputation_output,
            imputation_output
        )

    
    def _fineTuning(self, x_enc, time_mark, valid_mask=None, next_x_enc=None, mode='all'):
        pass

    def _predict(self, x_enc, time_mark, valid_mask=None, mode='all'):
        pass

    def forward(self, x_enc, time_mark=None, valid_mask=None, next_x_enc=None, mode='train'):
       
        """General forward pass."""
        # print('model inference...')
        if mode == 'train':
            return self._train(x_enc, time_mark, valid_mask, next_x_enc, self.configs.traning_mode)
        elif mode == 'fine-tune':
            return self._fineTuning(x_enc, time_mark, valid_mask, next_x_enc, self.configs.fine_tune_mode)  
        elif mode == 'pred' or mode == 'test':
            return self._predict(x_enc, time_mark, valid_mask)
        else:
            raise ValueError(f"Supported mode: train or predict")
        # else:
        #     return self._predict(x_enc, time_mark, valid_mask, next_x_enc)



def imputation_loss(pred, target, mask, alpha, beta):
    rec_loss = mse_loss(pred, target, mask)

    smo_loss = smooth_loss(pred, mode='dy2')

    return alpha * rec_loss + beta * smo_loss