import torch
import torch.nn as nn
import einops
import numpy as np
import math
from utils.losses import mse_loss, smooth_loss, nll_loss, huber_loss, nll_loss_student_t, calculate_nll_t_per_timestep, \
    calculate_nll_per_timestep, dispersive_loss
from utils.tools import apply_mask, process_attention, apply_mask_seasons, sequence2seasons
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from layers.Self_layers import make_2tuple, Swish, RotaryPositionEmbedding1D, PatchEmbedding, ResidualVectorQuantizer, InterSeasonBlock, IntraSeasonBlock, UpsampleBlock, DownsampleBlock


class Encoder(nn.Module):
    def __init__(
        self,
        bands: int,
        dim: int = 64,
        num_heads: int = 4,
        num_groups: int = 2,
        num_register_tokens: int = 4,
        mlp_ratio: float = 2.0,
        order: list = ["inter", "intra"],
        dropout: float = 0.1
    ):
        super().__init__()
        self.steps_per_season = 31
        self.season = 12
        self.bands = bands
        self.dim = dim
        self.num_groups = num_groups
        self.order = order
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        # 注册令牌
        self.register_token = nn.Parameter(torch.randn(1, num_register_tokens, dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # 投影层：将输入通道数 (bands + 1) 投影到 dim
        self.proj = nn.Linear(bands + 1, dim)
        self.norm = nn.LayerNorm(dim)

        # 预先注册的填充掩码（由外部提供）
        self.register_buffer('padding_mask', None)
        self.register_buffer('intra_mask', None)
        self.register_buffer('inter_mask', None)

        # Transformer 块组
        self.groups = nn.ModuleList([
            nn.ModuleList([
                IntraSeasonBlock(dim, num_heads, num_register_tokens, mlp_ratio, dropout) if block_type == "intra"
                else InterSeasonBlock(dim, num_heads, num_register_tokens, mlp_ratio, dropout) for block_type in order
            ]) for _ in range(num_groups)
        ])
        self.intermediate_projection = nn.Linear(dim, dim)
        self.rope_time = RotaryPositionEmbedding1D(dim=dim, frequency=5000)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, list, tuple]:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, season, steps_per_season, bands]
            mask (torch.Tensor | None): 可选的掩码，形状为 [batch_size, season, steps_per_season]
        
        Returns:
            tuple: (
                输出张量 [batch_size, season * steps_per_season, dim],
                中间特征列表,
                注意力输出元组
            )
        """
        batch_size = x.shape[0]

        # 拼接 mask
        if mask is not None:
            # print('mask.shape', mask.shape)
            # print('x.shape', x.shape)
            mask = mask.unsqueeze(-1)  # [batch_size, season, steps_per_season, 1]
            x = torch.cat([x, mask], dim=-1)  # [batch_size, season, steps_per_season, bands + 1]
        else:
            x = x  # 不拼接 mask

        # 投影到嵌入维度
        x = self.proj(x)  # [batch_size, season, steps_per_season, dim]
        x = self.norm(x)

        # 应用旋转位置编码
        positions = torch.arange(self.season * self.steps_per_season, device=x.device)
        x = x.view(batch_size, -1, self.dim)  # [batch_size, season * steps_per_season, dim]
        x = self.rope_time(x, positions)
        x = x.view(batch_size, self.season, self.steps_per_season, self.dim)

        # 处理掩码
        # intra_mask = self.intra_mask
        # inter_mask = self.inter_mask
        # if mask is not None:
        #     if self.padding_mask is not None:
        #         mask = mask * self.padding_mask  # 结合外部传入的掩码和填充掩码
        #     intra_mask = (1 - mask.view(batch_size * self.season, self.steps_per_season)).to(x.device)
        #     inter_mask = (1 - mask.view(batch_size * self.steps_per_season, self.season)).to(x.device)

        # Transformer 块处理
        inter_features = []
        attn_outputs = []
        register_tokens = self.register_token
        for group in self.groups:
            for block_type, block in zip(self.order, group):
                if block_type == "intra":
                    x, attn_intra = block(x, batch_size, self.season, self.steps_per_season, register_tokens)
                    attn_outputs.append(("intra", attn_intra))
                elif block_type == "inter":
                    x, attn_inter = block(x, batch_size, self.season, self.steps_per_season, register_tokens)
                    attn_outputs.append(("inter", attn_inter))
                    inter_features.append(x.view(batch_size, -1))

        # 中间投影
        x = x.view(batch_size, -1, self.dim)
        x = self.intermediate_projection(x)

        return x, inter_features, tuple(attn_outputs)


class ImputationHead(nn.Module):
    def __init__(self, steps, bands, season, dim=64, num_heads=4, num_groups=1, num_register_tokens=4, mlp_ratio=2, dropout=0.1,
                 steps_per_season=None, padding=0):
        super().__init__()
        self.steps = steps  # Original steps without padding
        self.bands = bands
        self.season = season
        self.dim = dim
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        self.padding = padding
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.register_token = nn.Parameter(torch.randn(1, num_register_tokens, dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # Total steps including padding
        self.total_steps = steps + padding
        # Use provided steps_per_season or calculate it
        self.steps_per_season = steps_per_season if steps_per_season is not None else (steps + padding + season - 1) // season
        self.total_steps = self.steps_per_season * season  # Adjust total_steps to be divisible by season

        self.groups = nn.ModuleList([
            nn.ModuleList([
                InterSeasonBlock(dim, num_heads, num_register_tokens, mlp_ratio, dropout)
            ]) for _ in range(num_groups)
        ])
        self.output_projection = nn.Linear(dim, bands)
        self.log_sigma_head = nn.Linear(dim, bands)
        self.rope_time = RotaryPositionEmbedding1D(dim=dim, frequency=5000)

    def forward(self, x):
        batch_size = x.size(0)
        # Input shape: (batch_size, total_steps, dim)
        assert x.shape[1] == self.total_steps, f"Expected {self.total_steps} steps, got {x.shape[1]}"
        # Reshape to (batch_size, season, steps_per_season, dim)
        x = x.reshape(batch_size, self.season, self.steps_per_season, self.dim).contiguous()
        register_tokens = self.register_token
        for group in self.groups:
            for block in group:
                x, _ = block(x, batch_size, self.season, self.steps_per_season, register_tokens, mask=None)
        x = x.reshape(batch_size, -1, self.dim)
        output = self.output_projection(x)  # Shape: (batch, total_steps, bands)
        log_sigma = self.log_sigma_head(x)
        # Remove padding to match original steps
        if self.padding > 0:
            output = output[:, :self.steps, :]
            log_sigma = log_sigma[:, :self.steps, :]
        return output, log_sigma


class ForecastHead(nn.Module):
    def __init__(self, steps, bands, season, dim=64, num_heads=4, num_groups=1, num_register_tokens=4, mlp_ratio=2, dropout=0.1,
                 steps_per_season=None, padding=0):
        super().__init__()
        self.steps = steps  # Original steps without padding
        self.bands = bands
        self.season = season
        self.dim = dim
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        self.padding = padding
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.register_token = nn.Parameter(torch.randn(1, num_register_tokens, dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # Total steps including padding
        self.total_steps = steps + padding
        # Use provided steps_per_season or calculate it
        self.steps_per_season = steps_per_season if steps_per_season is not None else (steps + padding + season - 1) // season
        self.total_steps = self.steps_per_season * season  # Adjust total_steps to be divisible by season

        self.groups = nn.ModuleList([
            nn.ModuleList([
                InterSeasonBlock(dim, num_heads, num_register_tokens, mlp_ratio, dropout, use_causal_attention=True)
            ]) for _ in range(num_groups)
        ])
        self.output_projection = nn.Linear(dim, bands)
        self.log_sigma_head = nn.Linear(dim, bands)
        self.rope_time = RotaryPositionEmbedding1D(dim=dim, frequency=5000)

    def forward(self, x):
        batch_size = x.size(0)
        # Input shape: (batch_size, total_steps, dim)
        assert x.shape[1] == self.total_steps, f"Expected {self.total_steps} steps, got {x.shape[1]}"
        # Reshape to (batch_size, season, steps_per_season, dim)
        x = x.reshape(batch_size, self.season, self.steps_per_season, self.dim).contiguous()

        register_tokens = self.register_token
        for group in self.groups:
            for block in group:
                x, _ = block(x, batch_size, self.season, self.steps_per_season, register_tokens)
        x = x.reshape(batch_size, -1, self.dim)
        output = self.output_projection(x)  # Shape: (batch, total_steps, bands)
        log_sigma = self.log_sigma_head(x)
        # Remove padding to match original steps
        if self.padding > 0:
            output = output[:, :self.steps, :]
            log_sigma = log_sigma[:, :self.steps, :]
        return output, log_sigma


class AnomalyHead(nn.Module):
    def __init__(self, dim, out_dim=5):
        super().__init__()
        # 第一层MLP：降维并添加非线性
        self.hidden_dim = dim//2
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU()
        )
        # 第二层MLP：输出均值
        self.mean_head = nn.Linear(self.hidden_dim, out_dim)
        # 第三层MLP：输出log_sigma
        self.log_sigma_head = nn.Linear(self.hidden_dim, out_dim)
        # 残差连接的投影层（匹配输入维数到hidden_dim）
        self.residual_proj = nn.Linear(dim, self.hidden_dim)

    def forward(self, x):
        # 输入 x: (batch_size, total_steps, dim)
        residual = self.residual_proj(x)  # Shape: (batch_size, total_steps, hidden_dim)
        x = self.mlp1(x)  # Shape: (batch_size, total_steps, hidden_dim)
        x = x + residual  # 残差连接
        mean = self.mean_head(x)  # Shape: (batch_size, total_steps, 1)
        log_sigma = self.log_sigma_head(x)  # Shape: (batch_size, total_steps, 1)
        return mean, log_sigma  # 每个时间步的均值和log_sigma


class Autoencoder(nn.Module):
    def __init__(self, steps, bands, season, dim=64, num_heads=4, num_groups=6, head_groups=2, mlp_ratio=2, num_register_tokens=4,
                 order=["inter", "intra"], dropout=0.1):
        super().__init__()
        self.steps = steps
        self.bands = bands
        self.season = season
        self.dim = dim

        # Encoder
        self.encoder = Encoder(
            bands=bands,
            dim=dim,
            num_heads=num_heads,
            num_groups=num_groups,
            num_register_tokens=num_register_tokens,
            mlp_ratio=mlp_ratio,
            order=order,
            dropout=dropout
        )

        self.vq = ResidualVectorQuantizer(num_embeddings=256, embedding_dim=32, commitment_cost=0.25, num_codebooks=6)
        self.pre_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=32,
            kernel_size=1,
            stride=1,
        )
        self.aft_conv = nn.Conv1d(
            in_channels=32,
            out_channels=dim,
            kernel_size=1,
            stride=1,
        )

        # Calculate the output steps after encoder
        # self.steps_per_season = self.encoder.steps_per_season
        self.padding = 6
        self.total_steps = 372


        # Downsampling layers as ModuleList
        self.downsample_blocks = nn.ModuleList([
            DownsampleBlock(
                in_channels=dim,
                out_channels=dim,
                kernel_size=4,
                stride=4,
                padding=0
            ),
            DownsampleBlock(
                in_channels=dim,
                out_channels=dim,
                kernel_size=4,
                stride=4,
                padding=0
            )
        ])

        # Upsampling layers as ModuleList
        self.downsample1_steps = (self.total_steps - 4) // 4 + 1
        self.downsample2_steps = (self.downsample1_steps - 4) // 4 + 1
        self.upsample_blocks = nn.ModuleList([
            UpsampleBlock(
                in_channels=dim,
                out_channels=dim,
                kernel_size=4,
                stride=4,
                padding=0,
                output_padding=1 if self.downsample1_steps * 4 != self.total_steps else 0
            ),
            UpsampleBlock(
                in_channels=dim,
                out_channels=dim,
                kernel_size=4,
                stride=4,
                padding=0,
                output_padding=0
            )
        ])

        self.extra_padding = self.total_steps - (self.downsample2_steps * 4 * 4)

        # Imputation and Forecast heads
        self.imputation_head = ImputationHead(
            steps=steps,
            bands=bands,
            season=season,
            dim=dim,
            num_heads=num_heads,
            num_groups=head_groups,
            num_register_tokens=num_register_tokens,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            steps_per_season=31,
            padding=self.padding
        )
        self.forecast_head = ForecastHead(
            steps=steps,
            bands=bands,
            season=season,
            dim=dim,
            num_heads=num_heads,
            num_groups=head_groups,
            num_register_tokens=num_register_tokens,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            steps_per_season=31,
            padding=self.padding
        )
        self.anomaly_head = AnomalyHead(
            dim,
            out_dim=bands
        )

    def encode(self, x, mask=None, time_mark=None):
        # Encoder
        x, inter_features, attn_outputs = self.encoder(x, mask)  # Shape: (batch, steps + padding, dim)
        x = x.permute(0, 2, 1)  # Shape: (batch, dim, steps + padding)
        # print(x.shape)
        # Downsampling
        for block in self.downsample_blocks:
            x = block(x)  # Apply downsample -> norm -> silu
        x = self.pre_conv(x)
        return x, x.clone(), attn_outputs

    def decode(self, x, quantized):
        x = self.aft_conv(x)  # Shape: (batch, dim, downsample2_steps)
        quantized = self.aft_conv(quantized)  # Shape: (batch, dim, downsample2_steps)
        # Upsampling
        for block in self.upsample_blocks:
            x = block(x)  # Apply upsample -> norm -> silu
        for block in self.upsample_blocks:
            quantized = block(quantized)  # Apply upsample -> norm -> silu
        # Pad to match total_steps
        if self.extra_padding > 0:
            x = F.pad(x, (0, self.extra_padding), mode = 'constant', value = 0)
            quantized = F.pad(quantized, (0, self.extra_padding), mode='constant', value=0)
        # Select head
        x = x.permute(0, 2, 1)  # Shape: (batch, total_steps, dim)
        quantized = quantized.permute(0, 2, 1)  # Shape: (batch, total_steps, dim)

        imputation_x, imputation_log_sigma = self.imputation_head(x)

        forecast_x, forecast_log_sigma = self.forecast_head(x)

        anomaly_mean, anomaly_log_sigma = self.anomaly_head(quantized)

        return {
            'imputation': (imputation_x,imputation_log_sigma),
            'forecast': (forecast_x,forecast_log_sigma),
            'anomaly': (anomaly_mean, anomaly_log_sigma)
        }

    def forward(self, x, mask=None, time_mark=None):

        # Encode
        x, inter_features, attn_outputs = self.encode(x, mask=mask, time_mark=time_mark)  # Shape: (batch, dim, downsample2_steps)

        # z = self.pre_conv(x)  # Shape: (batch, 32, downsample2_steps)
        vq_loss, quantized, perplexity, all_encodings = self.vq(x)
        # print('vq', vq_loss)
        # print('entropy loss', entropy_loss)
        # Decode
        output = self.decode(x, quantized)  # Shape: (batch, steps, bands)
        return output, vq_loss, inter_features, attn_outputs


# def prepare_data(x, season):
#     """
#     将输入数据 (batch, steps, bands) 转换为 (batch, season, steps//season, bands)
#     """
#     batch, steps, bands = x.shape
#     steps_per_season = steps // season
#     if steps % season != 0:
#         raise ValueError("steps must be divisible by season")
#     x = einops.rearrange(x, 'b (s t) d -> b s t d', s=season, t=steps_per_season)
#     return x


def reverse_data(x, steps):
    """
    将数据 (batch, season, steps//season, bands) 转换回 (batch, steps, bands)
    """
    x = einops.rearrange(x, 'b s t d -> b (s t) d', s=x.size(1), t=x.size(2))
    if x.size(1) != steps:
        raise ValueError(f"Output steps ({x.size(1)}) do not match expected steps ({steps})")
    return x


def imputation_loss(pred, target, mask, alpha, beta):
    rec_loss = mse_loss(pred, target, mask)

    smo_loss = smooth_loss(pred, mode='dy2')

    return alpha * rec_loss + beta * smo_loss

def forecast_loss(pred, target, mask, alpha, beta):
    rec_loss = mse_loss(pred, target, mask)

    smo_loss = smooth_loss(pred, mode='dy2')

    return alpha * rec_loss + beta * smo_loss

def anomaly_loss(pred, log_sigma, target, mask, vq_loss):
    rec_loss = mse_loss(pred, target, mask)
    uncertainty_loss = nll_loss(pred, log_sigma, target, mask)

    return rec_loss + uncertainty_loss + vq_loss

def cal_rec_loss(pred, target, mask, alpha, beta):
    rec_loss = mse_loss(pred, target, mask)

    smo_loss = smooth_loss(pred, mode='dy2')

    return alpha * rec_loss + beta * smo_loss

def cal_nll_loss(mu, log_sigma, labels, mask):
    return nll_loss(mu, log_sigma, labels, mask)


def cal_nll_t(mu, log_sigma, df_raw, labels, mask):
    return nll_loss_student_t(mu, log_sigma, df_raw, labels, mask)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.autoencoder = Autoencoder(steps=configs.seq_len, bands=configs.enc_in, season=12, 
        dim=configs.d_model, num_heads=configs.n_heads, num_groups=configs.e_layers, 
        head_groups=configs.d_layers, mlp_ratio=configs.mlp_ratio, num_register_tokens=configs.num_register_tokens, dropout=configs.dropout)
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
        
        x_processed, padding_mask, valid_mask = sequence2seasons(x_enc, season=12)
        # print('x_processed.shape', x_processed.shape)
        # print('valid_mask.shape', valid_mask.shape)

        # Apply mask
        ori_batch_x, batch_x, missing_mask, indicating_mask = apply_mask_seasons(
            ori_batch_x=x_processed,
            ori_valid_mask=valid_mask,
            p=mask_ratio,
            device='cuda',
            mode='train'
        )
        # print('ori_batch_x.shape', ori_batch_x.shape)
        # print('batch_x.shape', batch_x.shape)
        # print('valid_mask.shape', valid_mask.shape)
        # print('missing_mask.shape', missing_mask.shape)
        # print('indicating_mask.shape', indicating_mask.shape)   
        return ori_batch_x, batch_x, valid_mask, missing_mask, indicating_mask


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
        batch_x = batch_x.view(batch_x.size(0), -1, batch_x.size(3))[:,:366,:]
        indicating_mask = indicating_mask.view(indicating_mask.size(0), -1, indicating_mask.size(3))[:,:366,:]
        
        # time_mark = self.dropout(time_mark)
        # ViT 前向传播，获取输出、vq_loss 和 inter_features
        output, vq_loss, inter_features, attn_outputs = self.autoencoder(batch_x_masked, mask=missing_mask.float())
       
        missing_mask = missing_mask.view(missing_mask.size(0), -1)[:,:366]
    
        # 根据模式计算损失
        if mode == 'imputation':
            # 填补模式：重建原始输入
            pred = output['imputation'][0][:, :366, :]
            log_sigma = output['imputation'][1][:, :366, :]
            # print('batch_x.shape', batch_x.shape)
            loss = cal_rec_loss(pred, batch_x, indicating_mask, 1, 0.5)
            # print(loss)
        elif mode == 'forecast':
            # 预测模式：预测未来时间步
            if next_x_enc is None:
                pred = 0
                loss = 0
            else:
                next_x, next_valid_mask = self._prepare_forecast_label(next_x_enc)
                pred = output['forecast'][0][:, :366, :]  # 假设 forecast 输出存在
                log_sigma = output['forecast'][1][:, :366, :]
                loss = cal_rec_loss(pred, next_x[:, :366, :], next_valid_mask, 1, 0.5)

        elif mode == 'anomaly':
            # 异常检测模式：重建输入并可选计算异常概率
            pred = output['anomaly'][0][:, :366, :]
            log_sigma = output['anomaly'][1][:, :366, :]
            loss = cal_rec_loss(pred, batch_x, missing_mask, 1, 0.5) + vq_loss
        elif mode == 'all':
            pred=0
            # 综合模式：同时优化填补、预测和异常检测任务
            loss = 0
            # 填补任务损失
            if 'imputation' in output:
                pred_imp = output['imputation'][0][:, :366, :]
                loss += cal_rec_loss(pred_imp, batch_x, indicating_mask, 1, 0.5)
            # 预测任务损失
            if 'forecast' in output and next_x_enc is not None:
                next_x, next_valid_mask = self._prepare_forecast_label(next_x_enc)
                pred_fc = output['forecast'][0][:, :366, :]  # 假设 forecast 输出存在
                loss = cal_rec_loss(pred_fc, next_x[:, :366, :], next_valid_mask, 1, 0.5)
            # 异常检测任务损失
            if 'anomaly' in output:
                pred_anom = output['anomaly'][0][:, :366, :]
                log_sigma = output['anomaly'][1][:, :366, :]
                loss += cal_rec_loss(pred_anom, batch_x, missing_mask, 1, 0.5) + vq_loss
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if self.configs.dispersive_loss == 'True':
            disp_loss = dispersive_loss(inter_features)
            loss += disp_loss

        # 返回预测、总损失和输出
        return (
            output['imputation'][0][:, :366, :],
            loss,
            output['anomaly'][0][:, :366, :] if 'anomaly' in output else None,
            output['imputation'][0][:, :366, :] if 'imputation' in output else None
        )
    
    def _fineTuning(self, x_enc, time_mark, valid_mask=None, next_x_enc=None, mode='all'):
        """
        VAE 训练的前向传播，根据随机化的 mask_rate 优化 imputation 或 forecast 和 anomaly 任务。

        参数：
            x_enc: 输入编码，形状为 (batch_size, seq_len, feature_dim)
            time_mark: 时间标记
            valid_mask: 有效掩码，默认为 None
            next_x_enc: 预测模式下的目标数据，默认为 None

        返回：
            pred: 模型预测输出
            loss: 总损失（包括重建损失、vq_loss 和 Dispersive Loss）
            anomaly_output: 异常模式输出（用于后续处理）
            imputation_output: 填补模式输出（用于后续处理）
        """

        # 随机化 mask_rate
        mask_rate = 0 if np.random.rand() < 0.5 else np.random.uniform(0, 0.8)

        # 准备输入
        batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
            x_enc, mask_rate, valid_mask
        )

        time_mark = self.dropout(time_mark)
        # ViT 前向传播，获取输出、vq_loss 和 inter_features
        output, vq_loss, inter_features, attn_outputs = self.autoencoder(batch_x_masked, mask=missing_mask[:, :, 0].float(), time_mark=time_mark)

        # 初始化返回变量
        pred = 0
        loss = 0

        # 根据 mask_rate 计算损失
        if mask_rate == 0:
            # mask_rate 为 0，优化 forecast 和 anomaly detection
            if 'forecast' in output and next_x_enc is not None:
                next_x, next_valid_mask = self._prepare_forecast_label(next_x_enc)
                pred_fc = output['forecast'][0][:, :366, :]
                log_sigma_fc = output['forecast'][1][:, :366, :]
                forecast_mask = next_valid_mask.clone()  # 复制以避免原地修改
                forecast_mask[:, :-31, :] = 0  # 只保留最后 31 步
                loss += cal_rec_loss(pred_fc, next_x[:, :366, :], next_valid_mask, 1, 0.5) + 0.2 * nll_loss(pred_fc, log_sigma_fc, next_x[:, :366, :], forecast_mask)
            if 'anomaly' in output:
                pred_anom = output['anomaly'][0][:, :366, :]
                log_sigma_anom = output['anomaly'][1][:, :366, :]
                loss += cal_rec_loss(pred_anom, batch_x, missing_mask, 1, 0.5) + 0.2 * nll_loss(pred_anom, log_sigma_anom, batch_x[:, :366, :], missing_mask)
        else:
            # mask_rate 不为 0，仅优化 imputation
            if 'imputation' in output:
                pred_imp = output['imputation'][0][:, :366, :]
                log_sigma_imp = output['imputation'][1][:, :366, :]
                loss += cal_rec_loss(pred_imp, batch_x, indicating_mask, 1, 0.5) + 0.2 * nll_loss(pred_imp, log_sigma_imp, batch_x[:, :366, :], indicating_mask)

        return (
            output['imputation'][0][:, :366, :],
            loss,
            output['anomaly'][0][:, :366, :] if 'anomaly' in output else None,
            output['imputation'][0][:, :366, :] if 'imputation' in output else None
        )


    def _predict(self, x_enc, time_mark, valid_mask=None, mode='all'):
        """
        VAE 预测的前向传播，支持 imputation、forecast 和 anomaly 模式。

        参数：
            x_enc: 输入编码，形状为 (batch_size, seq_len, feature_dim)
            time_mark: 时间标记
            valid_mask: 有效掩码，默认为 None
            mode: 预测模式，'imputation', 'forecast' 或 'anomaly'

        返回：
            pred: 模型预测输出
            anomaly_output: 异常模式输出（用于后续处理）
            imputation_output: 填补模式输出（用于后续处理）
        """
        # 准备输入
        batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
            x_enc, self.configs.mask_rate, valid_mask
        )

        time_mark = self.dropout(time_mark)
        # ViT 前向传播，获取输出、vq_loss 和 inter_features
        output, vq_loss, inter_features, attns = self.autoencoder(batch_x_masked, mask=missing_mask.float())

        # 根据模式选择预测输出
        if mode == 'imputation':
            # 填补模式：返回填补输出
            pred = output['imputation'][0][:, :366, :]
            anomaly_output = None
            imputation_output = pred
        elif mode == 'forecast':
            # 预测模式：返回预测输出
            pred = output['forecast'][0][:, :366, :] if 'forecast' in output else None
            anomaly_output = None
            imputation_output = None
        elif mode == 'anomaly':
            # 异常检测模式：返回异常检测输出
            pred = output['anomaly'][0][:, :366, :] if 'anomaly' in output else None
            anomaly_output = pred
            imputation_output = None
        elif mode == 'all':
            # 综合模式：返回所有相关输出
            # pred = {
            #     'imputation': output['imputation'][:, :366, :] if 'imputation' in output else None,
            #     'forecast': output['forecast'][:, :366, :] if 'forecast' in output  else None,
            #     'anomaly': output['anomaly'][0][:, :366, :] if 'anomaly' in output else None
            # }
            imputation_output = output['imputation'][0][:, :366, :]
            # nll = nll_t_per_step(output['anomaly'][0][:, :366, :], output['anomaly'][1][:, :366, :], output['anomaly'][2], batch_x, valid_mask)
            # anomalies = z_score_detector(nll[:,:,0], valid_mask[:,:,0])
        else:
            raise ValueError(f"Unknown mode: {mode}")
        inter_attns = process_attention(attns, register_tokens=4, seasons=12, steps_per_season=31)
        # print('inter_features.shape', inter_features.shape)
        return output['imputation'][0][:, :366, :], inter_features.permute(0,2,1), inter_attns, imputation_output

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


