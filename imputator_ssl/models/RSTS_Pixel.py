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
        
        # Concatenate input with time_mark and apply patch embedding
        x = torch.cat([x, time_mark], dim=-1)  # [B, steps, bands+2]
        x = self.patch_embed(x)  # [B, num_patches, kernel, dim]
        
        # Generate masks
        intra_mask, inter_mask = self._get_masks(B, mask)
        
        # Intra-season attention
        x = self._add_register_tokens(x, self.intra_register_token, dim=2)  # [B, num_patches, kernel + num_register_tokens, dim]
        x, _ = self.intra_attn(x, mask=intra_mask)  # [B, num_patches, kernel + num_register_tokens, dim]
        x = self._remove_register_tokens(x, dim=2, keep_cls=False)  # [B, num_patches, kernel, dim]
        
        # inter-season attention
        x = self._add_register_tokens(x, self.inter_register_token, dim=1)  # [B, num_patches * kernel + num_register_tokens, dim]
        x, _ = self.inter_attn(x, mask=inter_mask)  # [B, num_patches * kernel + num_register_tokens, dim]
        x = self._remove_register_tokens(x, dim=1, keep_cls=False)  # [B, num_patches * kernel, dim]
        
        return x, intra_mask, inter_mask
    
    def _get_masks(self, B, mask=None):
        """ register token """
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


class ImputationHead(nn.Module):
    def __init__(self, dim, total_steps, bands):
        super().__init__()
        self.head = nn.Linear(dim, bands)
        self.total_steps = total_steps
    
    def forward(self, x):
        # [B, P, K, D] -> [B, P*K, D] -> [B, P*K, bands]
        B, P, K, D = x.shape
        x = x.reshape(B, -1, D)  # 直接flatten
        x = self.head(x)         
        return x[:, :self.total_steps, :]  

class ResidualBlock1D(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=pad, dilation=dilation),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size, padding=pad, dilation=dilation),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, dim, L)
        out = self.net(x) + x  # residual
        out = out.permute(0, 2, 1)  # (B, L, dim)
        out = self.norm(out)
        out = out.permute(0, 2, 1)  # (B, dim, L)
        return out

class VQVAEHead(nn.Module):
    def __init__(self, dim, out_dim=5, num_embeddings=256, embedding_dim=32,
                 commitment_cost=0.25, num_codebooks=4,
                 downsample_factor=8, decoder_blocks=2):  # 新增：解码器块数
        super().__init__()
        self.downsample_factor = downsample_factor
        self.decoder_blocks = decoder_blocks

        # 下采样
        self.down_conv = nn.Conv1d(dim, embedding_dim, kernel_size=downsample_factor,
                                   stride=downsample_factor, padding=0)

        # 上采样
        self.up_conv = nn.ConvTranspose1d(embedding_dim, dim, kernel_size=downsample_factor,
                                          stride=downsample_factor, padding=0)

        # === 新增：残差解码块 ===
        self.decoder_blocks_list = nn.ModuleList([
            ResidualBlock1D(dim) for _ in range(decoder_blocks)
        ])

        self.vq = ResidualVectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            num_codebooks=num_codebooks
        )

        # 预测头
        self.mean_head = nn.Linear(dim, out_dim)
        self.log_sigma_head = nn.Linear(dim, out_dim)


    def forward(self, x):
        B, S, T, D = x.shape
        L = S * T
        # print('x shape:', x.shape)
        x_flat = x.reshape(B, L, D)
        x_proj = x_flat.permute(0, 2, 1)          # (B, D, L)
        # print('x_proj shape:', x_proj.shape)
        # 下采样
        z_e = self.down_conv(x_proj)              # (B, embedding_dim, L//k)
        z_e = z_e.permute(0, 2, 1)                # (B, L//k, embedding_dim)
        # print('z_e shape:', z_e.shape)
        # VQ
        vq_loss, z_q, perplexity, encodings = self.vq(z_e)
        z_q = z_q.permute(0, 2, 1)                 # (B, embedding_dim, L//k)
        # print('z_q shape:', z_q.shape)
        # 上采样
        decoded = self.up_conv(z_q)               # (B, dim, L)
        # print('decoded shape:', decoded.shape)
        # === 新增：残差解码块（增强重建能力）===
        for block in self.decoder_blocks_list:
            decoded = block(decoded)              # (B, dim, L)
        # print('decoded shape:', decoded.shape)
        decoded = decoded.permute(0, 2, 1)         # (B, L, dim)
        # print('decoded shape:', decoded.shape)
        # 预测
        mean = self.mean_head(decoded)            # (B, L, out_dim)
        log_sigma = self.log_sigma_head(decoded)

        return {
            'mean': mean,
            'log_sigma': log_sigma,
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'encodings': encodings,
            'z_q': z_q,
        }

class ForecastHead(nn.Module):
    def __init__(self, seq_len, dim, forecast_length, bands,
                 mlp_hidden=128, mlp_layers=2, dropout=0.1):
        """
        seq_len: 输入序列长度 (e.g., 48)
        dim: 输入特征维度 (e.g., 64)
        forecast_length: 预测步数 (e.g., 24)
        bands: 输出波段数 (e.g., 32)
        """
        super().__init__()
        self.forecast_length = forecast_length
        self.bands = bands

        # === Step 1: MLP - 每个时间步 dim → bands ===
        layers = []
        in_features = dim
        for i in range(mlp_layers):
            out_features = mlp_hidden if i < mlp_layers - 1 else bands
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(out_features) if i == mlp_layers - 1 else nn.Identity()
            ])
            in_features = out_features
        self.mlp = nn.Sequential(*layers)

        # === Step 2: Linear - 序列外推 seq_len → forecast_length ===
        self.seq_linear = nn.Linear(seq_len, forecast_length)


    def forward(self, x):
        """
        x: (B, S, T, D)
        返回: (B, forecast_length, bands)
        """
        B, S, T, D = x.shape
        L = S * T

        x_flat = x.reshape(B, L, D)

        # === 1. MLP: 逐时间步变换 dim → bands ===
        x_mlp = self.mlp(x_flat)                     # (B, seq_len, bands)

        # === 2. 转置 + Linear: 序列外推 ===
        x_T = x_mlp.transpose(1, 2)             # (B, bands, seq_len)
        pred_T = self.seq_linear(x_T)           # (B, bands, forecast_length)
        pred = pred_T.transpose(1, 2)           # (B, forecast_length, bands)


        return pred  # (B, forecast_length, bands)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.encoder = Encoder(steps=configs.seq_len, bands=configs.enc_in, steps_per_season=14, 
        dim=configs.d_model, num_heads=configs.n_heads, num_groups=6, mlp_ratio=configs.mlp_ratio, num_register_tokens=configs.num_register_tokens, dropout=configs.dropout)
        self.imputation_head = ImputationHead(dim=configs.d_model, total_steps=configs.seq_len, bands=configs.enc_in)
        self.vqvae_head = VQVAEHead(dim=configs.d_model, out_dim=configs.enc_in, num_embeddings=256, embedding_dim=32,
                                    commitment_cost=0.25, num_codebooks=6)
        # self.forecast_head = ForecastHead(seq_len=378, dim=configs.d_model, forecast_length=122, bands=configs.enc_in)
        self.configs = configs
        self.dropout = nn.Dropout(p=0.2)
        print('mask ratio:', configs.mask_rate)
        print('traning mode:', self.configs.traning_mode)

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

        # if next_x_enc is not None:
        #      print('next_x_enc shape:', next_x_enc.shape)


        # 准备输入
        batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
            x_enc, self.configs.mask_rate, valid_mask
        )

        time_mark = self.dropout(time_mark)
        # ViT 前向传播，获取输出、vq_loss 和 inter_features
        output, inter_features, attn_outputs = self.encoder(batch_x_masked, mask=missing_mask[:, :, 0].float(), time_mark=time_mark)
        imputation_output = self.imputation_head(output)
        vq_output = self.vqvae_head(output)
        # forecast_output = self.forecast_head(output)

        vq_loss = vq_output['vq_loss']
        loss = imputation_loss(imputation_output, batch_x, indicating_mask, alpha=0.8, beta=0.2)

        # if mode == 'all':
        #     if next_x_enc is None:
        #         pass
        #         # pred = 0
        #         # loss = 0
        #         # next_x, next_valid_mask = self._prepare_forecast_label(next_x_enc)
        #         # forecast_loss = imputation_loss(forecast_output, next_x, next_valid_mask, 0.8, 0.2)
        #         # loss += forecast_loss
        #     vq_head_loss = imputation_loss(vq_output['mean'][:, :self.configs.seq_len, :], batch_x, missing_mask, 1, 0) + vq_loss 
        #     # + 0.2*nll_loss(vq_output['mean'][:, :self.configs.seq_len, :], vq_output['log_sigma'][:, :self.configs.seq_len, :], batch_x[:, :self.configs.seq_len, :], missing_mask)
        #     loss += vq_head_loss

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
        return self._train(x_enc, time_mark, valid_mask, next_x_enc, self.configs.traning_mode)
        # # print('model inference...')
        # if mode == 'train':
        #     return self._train(x_enc, time_mark, valid_mask, next_x_enc, self.configs.traning_mode)
        # elif mode == 'fine-tune':
        #     return self._fineTuning(x_enc, time_mark, valid_mask, next_x_enc, self.configs.fine_tune_mode)  
        # elif mode == 'pred' or mode == 'test':
        #     return self._predict(x_enc, time_mark, valid_mask)
        # else:
        #     raise ValueError(f"Supported mode: train or predict")
        # else:
        #     return self._predict(x_enc, time_mark, valid_mask, next_x_enc)



def imputation_loss(pred, target, mask, alpha, beta):
    rec_loss = mse_loss(pred, target, mask)

    smo_loss = smooth_loss(pred, mode='dy2')

    return alpha * rec_loss + beta * smo_loss