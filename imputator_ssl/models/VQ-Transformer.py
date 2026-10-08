import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dask.array import zeros_like
from models.quant import VectorQuantizer2
from utils.losses import mse_loss, smooth_loss, nll_loss, huber_loss, nll_loss_student_t, calculate_nll_t_per_timestep, calculate_uncertainty_per_timestep, dispersive_loss
from utils.tools import apply_mask
from layers.Transformer_EncDec import AdaLN_EncoderLayer, AdaLN_Encoder, EncoderLayer, DiTEncoder, DiTEncoderLayer, init_adaLN_zero
from layers.Embedding import Embedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import random

def z_score_detector(anomaly_prob, mask, threshold=2.5):
    """
    基于z-score的无监督异常检测函数，根据有效数据比例动态调整严格程度。

    参数:
    - anomaly_prob: (batch, steps) 异常概率，0-1之间，越大表示异常概率越高。
    - mask: (batch, steps) 有效点mask，1为有效点，0为无效点。
    - threshold: z-score的基础阈值，默认值为3.0。

    返回:
    - anomaly_label: (batch, steps) 异常标注，1为异常，0为正常。
    """
    # 只对有效点进行计算
    valid_points = anomaly_prob * mask  # 有效点的异常概率
    valid_mask = mask  # 有效点mask

    # 计算有效点的均值和标准差
    count_valid = valid_mask.sum(dim=1, keepdim=True)
    mean = (valid_points.sum(dim=1, keepdim=True) / count_valid).nan_to_num()
    std = torch.sqrt(((valid_points - mean) ** 2 * valid_mask).sum(dim=1, keepdim=True) / count_valid).nan_to_num()

    # 计算z-score
    z_scores = (valid_points - mean) / (std + 1e-8)  # 防止除零

    # 计算每个样本的有效比例
    steps = mask.size(1)
    p = count_valid / steps  # 有效点数占比

    # 动态调整阈值：当有效比例<1/3时，使用对数函数增加阈值
    adjusted_threshold = torch.where(
        p < 1/3,
        threshold + torch.log(1.0 / (3 * p + 1e-8)),  # 对数平滑调整
        threshold
    )

    # 检测异常并应用mask
    anomaly_label = (z_scores.abs() > adjusted_threshold).float() * valid_mask

    return anomaly_label.unsqueeze(-1)  # 保持输出维度一致

class imputation4missing(nn.Module):
    def __init__(self, configs):
        super(imputation4missing, self).__init__()
        self.pred_len = configs.label_len
        self.output_attention = True
        self.d_model = configs.d_model


        # 嵌入层
        self.embedding = Embedding(
            d_in=configs.enc_in,
            d_model=configs.d_model,
            with_pos=True,
            embed_type=configs.embed,
            freq=configs.freq,
            n_max_steps=self.pred_len,
            dropout=0
        )

        # self.condition_mlp = nn.Sequential(
        #     nn.Linear(736, configs.d_model, bias=True),
        #     nn.SiLU(),
        #     nn.Linear(configs.d_model, configs.d_model, bias=True),
        # )

        self.condition_embed = nn.Linear(32, configs.d_model)
        self.mask_embed = nn.Linear(1, configs.d_model)
        self.time_embed = nn.Linear(2, configs.d_model)
        # Encoder with AdaLN
        self.encoder = DiTEncoder(
            [
                DiTEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, diag_mask_flag=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.label_len,
                    configs.d_ff,
                    dropout=configs.dropout,
                ) for l in range(configs.e_layers)
            ],
            nn.LayerNorm(configs.d_model)
        )

        self.output_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        # self.output_projection = final_layer(configs.d_model, configs.c_out)
        self.encoder.apply(init_adaLN_zero)

    def forward(self, x_enc, time_mark, missing_mask, vq):
        # x_enc: [batch, seq_len, enc_in]
        # x_mark_enc: [batch, seq_len, 3]
        # missing_mask: [batch, seq_len, 1]
        batch, seq, bands = x_enc.shape
        # 嵌入输入
        # enc_out = self.embedding(x_enc, missing_mask[:,:,0].unsqueeze(-1), x_mark_enc)  # [batch, seq_len, d_model]
        enc_out = self.embedding(x_enc)  # [batch, seq_len, d_model]


        # condition = self.condition_mlp(vq.view(batch, -1))
        vq_condition = self.condition_embed(vq.permute(0, 2, 1))
        vq_condition = F.interpolate(vq_condition.permute(0, 2, 1), size=366, mode='linear').permute(0, 2, 1)
        mask_embed = self.mask_embed(missing_mask)
        time_embed = self.time_embed(time_mark)

        condition = vq_condition

        # Encoder with AdaLN
        enc_out, attns = self.encoder(enc_out, condition)

        # 输出投影
        dec_out = self.output_projection(enc_out)  # [batch, seq_len, c_out]

        return dec_out

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
            Swish(),
            nn.Conv1d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            Swish(),
            nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv1d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 使用插值进行上采样（在时间维度上扩展2倍）
        x = F.interpolate(x, scale_factor=2.0, mode='linear', align_corners=False)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        # 在时间维度的右侧填充
        pad = (0, 1)  # 1D填充（左边0，右边1）
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class TemporalNonLocalBlock(nn.Module):
    def __init__(self, num_features):
        super(TemporalNonLocalBlock, self).__init__()
        self.num_features = num_features


        self.norm = nn.LayerNorm(num_features)

        # 1D卷积替代2D卷积（处理时序数据）
        self.q = nn.Conv1d(num_features, num_features, kernel_size=1)
        self.k = nn.Conv1d(num_features, num_features, kernel_size=1)
        self.v = nn.Conv1d(num_features, num_features, kernel_size=1)

        # 输出投影
        self.proj_out = nn.Conv1d(num_features, num_features, kernel_size=1)

    def forward(self, x):
        """
        输入x形状: (batch, features, steps)
        输出形状:   (batch, features, steps)
        """
        # 保存原始输入用于残差连接
        residual = x

        # 归一化并调整维度: (batch, features, steps) -> (batch, steps, features)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 计算Q, K, V
        q = self.q(x)  # (batch, features, steps)
        k = self.k(x)  # (batch, features, steps)
        v = self.v(x)  # (batch, features, steps)

        # 调整维度计算注意力
        batch, features, steps = q.shape

        # 计算注意力权重
        attn = torch.einsum('bft,bgs->bts', q, k)  # (batch, steps, steps)
        attn = attn * (features ** -0.5)  # 缩放
        attn = F.softmax(attn, dim=-1)  # 在最后一个维度归一化

        # 加权聚合
        out = torch.einsum('bts,bfs->bft', attn, v)  # (batch, features, steps)

        # 输出投影
        out = self.proj_out(out)

        # 残差连接
        return residual + out

class Encoder(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, embedding_dim=64, num_downsamples=4):
        super(Encoder, self).__init__()

        # 初始卷积层
        self.init_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 构建编码层
        self.layers = nn.ModuleList()
        current_channels = hidden_channels
        self.swish = Swish()

        for i in range(num_downsamples):
            # Only double channels every two downsampling steps (i.e., when i is odd: 1, 3, ...)
            if i % 2 == 1:
                out_channels = current_channels * 2
            else:
                out_channels = current_channels  # Keep channels the same

            self.layers.append(nn.ModuleList([
                ResidualBlock(current_channels, current_channels),  # Keep channel count
                DownSampleBlock(current_channels),  # Downsample time dimension by 2x
                ResidualBlock(current_channels, out_channels)  # Change channels (same or double)
            ]))

            current_channels = out_channels

        # 最终特征处理
        self.final_residual_1 = ResidualBlock(current_channels, current_channels)
        self.atten = TemporalNonLocalBlock(current_channels)
        self.final_residual_2 = ResidualBlock(current_channels, current_channels)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=current_channels)
        # 投影到嵌入空间
        self.pre_vq_conv = nn.Conv1d(
            in_channels=current_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        # 初始特征提取
        x = self.init_conv(x)
        # print('enc conv1', x.shape)

        # 通过编码层
        for residual1, downsample, residual2 in self.layers:
            x = residual1(x)  # Keep channel count
            x = downsample(x)  # Time dimension downsample (2x)
            x = residual2(x)  # Change channel count (same or double)
        # print('enc conv', x.shape)
        # 最终特征处理
        x = self.final_residual_1(x)
        # print('enc final_residual', x.shape)
        x = self.atten(x)
        x = self.final_residual_2(x)
        # print('enc final_residual1', x.shape)
        x = self.norm(x)
        x = self.swish(x)
        # 投影到嵌入空间
        z = self.pre_vq_conv(x)

        return z

class Decoder(nn.Module):
    def __init__(self, out_channels=4, hidden_channels=64, embedding_dim=64, num_upsamples=4):
        super(Decoder, self).__init__()

        # 计算初始通道数，与 Encoder 的最终通道数匹配
        self.init_channels = hidden_channels * (2 ** (num_upsamples // 2))  # 64 * 4 = 256

        # 初始卷积层：从 embedding_dim=64 升到 init_channels=256
        self.init_conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=self.init_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 初始特征处理，与 Encoder 的 final_residual_1, atten, final_residual_2 对称
        self.init_residual_1 = ResidualBlock(self.init_channels, self.init_channels)
        self.atten = TemporalNonLocalBlock(self.init_channels)
        self.init_residual_2 = ResidualBlock(self.init_channels, self.init_channels)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.init_channels)
        self.swish = Swish()

        # 构建解码层
        self.layers = nn.ModuleList()
        current_channels = self.init_channels  # 256

        for i in range(num_upsamples):
            # 每两个上采样阶段（i 为偶数时）通道数减半，对应 Encoder 的奇数阶段通道数翻倍
            if i % 2 == 0:
                out_channels_block = current_channels // 2
            else:
                out_channels_block = current_channels  # 保持通道数不变

            self.layers.append(nn.ModuleList([
                ResidualBlock(current_channels, current_channels),  # 保持通道数
                UpSampleBlock(current_channels),  # 时间维度上采样 2x
                ResidualBlock(current_channels, out_channels_block)  # 改变通道数
            ]))

            current_channels = out_channels_block

        # 最终输出层
        self.final_residual = ResidualBlock(current_channels, current_channels)
        self.rec_conv = nn.Conv1d(
            in_channels=current_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.log_sigma_conv = nn.Conv1d(
            in_channels=current_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.df_head = nn.Linear(current_channels, out_channels)
        nn.init.uniform_(self.df_head.weight, -0.01, 0.01)
        nn.init.constant_(self.df_head.bias, 0.0)
    def forward(self, z):
        # 初始特征转换
        x = self.init_conv(z)  # (batch_size, 64, steps//16) -> (batch_size, 256, steps//16)
        x = self.init_residual_1(x)
        x = self.atten(x)
        x = self.init_residual_2(x)
        x = self.norm(x)
        x = self.swish(x)

        # 通过解码层
        for residual1, upsample, residual2 in self.layers:
            x = residual1(x)  # 保持通道数
            x = upsample(x)  # 时间维度上采样 (2x)
            x = residual2(x)  # 改变通道数

        # 最终特征处理和输出
        x = self.final_residual(x)
        rec = self.rec_conv(x)  # (batch_size, 64, steps) -> (batch_size, 4, steps)
        log_sigma =  self.log_sigma_conv(x)
        df = self.df_head(x.permute(0,2,1).mean(dim=1))

        return rec.permute(0,2,1), log_sigma.permute(0,2,1), df.unsqueeze(1)


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=258, embedding_dim=32, commitment_cost=0.25, decay=0.99, eini=0.1,
                 num_codebooks=6):
        super(ResidualVectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._num_codebooks = num_codebooks

        # Create multiple codebooks
        self._embeddings = nn.ModuleList([
            nn.Embedding(self._num_embeddings, self._embedding_dim)
            for _ in range(self._num_codebooks)
        ])

        # Initialize embedding weights for each codebook
        for embedding in self._embeddings:
            if eini > 0:
                nn.init.trunc_normal_(embedding.weight.data, std=eini)
            else:
                embedding.weight.data.uniform_(-abs(eini) / self._num_embeddings, abs(eini) / self._num_embeddings)

        # EMA buffers for each codebook
        self.register_buffer('ema_vocab_hit', torch.zeros(self._num_codebooks, self._num_embeddings))
        self.register_buffer('ema_embedding', torch.stack([emb.weight.data.clone() for emb in self._embeddings]))
        self.register_buffer('record_hit', torch.tensor(0, dtype=torch.long))

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        residual = inputs.view(-1, self._embedding_dim)

        total_loss = 0
        quantized_all = torch.zeros_like(residual)
        all_encodings = []
        perplexities = []

        # Iterate through each codebook
        for i in range(self._num_codebooks):
            # Compute distances to current codebook
            distances = (torch.sum(residual ** 2, dim=1, keepdim=True) +
                         torch.sum(self._embeddings[i].weight ** 2, dim=1) -
                         2 * torch.matmul(residual, self._embeddings[i].weight.t()))

            # Find nearest embeddings
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)
            all_encodings.append(encoding_indices)

            # Quantize for this codebook
            quantized = torch.matmul(encodings, self._embeddings[i].weight)
            quantized_all += quantized

            # Compute losses
            e_latent_loss = F.mse_loss(quantized.detach(), residual)
            q_latent_loss = F.mse_loss(quantized, residual.detach())
            total_loss += q_latent_loss + self._commitment_cost * e_latent_loss

            # Update residual
            residual = residual - quantized.detach()

            # EMA updates during training
            if self.training:
                with torch.no_grad():
                    # Update usage counts
                    hit_V = encodings.sum(dim=0)
                    if self.record_hit == 0:
                        self.ema_vocab_hit[i].copy_(hit_V)
                    elif self.record_hit < 100:
                        self.ema_vocab_hit[i].mul_(0.9).add_(hit_V * 0.1)
                    else:
                        self.ema_vocab_hit[i].mul_(self._decay).add_(hit_V * (1 - self._decay))

                    # Update embeddings
                    dw = torch.matmul(encodings.t(), residual + quantized.detach())
                    self.ema_embedding[i].mul_(self._decay).add_(dw * (1 - self._decay))
                    # Normalize by cluster size
                    n = torch.sum(self.ema_vocab_hit[i])
                    ema_cluster_size = (self.ema_vocab_hit[i] + 1e-7) / (n + self._num_embeddings * 1e-7) * n
                    self._embeddings[i].weight.data.copy_(
                        self.ema_embedding[i] / (ema_cluster_size.unsqueeze(1) + 1e-7))

            # Compute perplexity for this codebook
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            perplexities.append(perplexity)

        if self.training:
            self.record_hit += 1

        # Straight-through estimator
        quantized_all = inputs.view(-1, self._embedding_dim) + (
                    quantized_all - inputs.view(-1, self._embedding_dim)).detach()
        quantized_all = quantized_all.view(input_shape)

        # Average perplexity across codebooks
        avg_perplexity = torch.mean(torch.stack(perplexities))
        return total_loss, quantized_all.permute(0, 2, 1).contiguous(), avg_perplexity
        # return total_loss, quantized_all.permute(0, 2, 1).contiguous(), avg_perplexity, all_encodings

# VectorQuantizer 模块（保持不变）
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=258, embedding_dim=32, commitment_cost=0.25, decay=0.99, eini=0.1):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay

        # Embedding layer
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)

        # Initialize embedding weights
        if eini > 0:
            nn.init.trunc_normal_(self._embedding.weight.data, std=eini)
        else:
            self._embedding.weight.data.uniform_(-abs(eini) / self._num_embeddings, abs(eini) / self._num_embeddings)

        # EMA buffers
        self.register_buffer('ema_vocab_hit', torch.zeros(self._num_embeddings))
        self.register_buffer('ema_embedding', self._embedding.weight.data.clone())
        self.register_buffer('record_hit', torch.tensor(0, dtype=torch.long))

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        # Compute distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) +
                     torch.sum(self._embedding.weight ** 2, dim=1) -
                     2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Find nearest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # EMA updates during training
        if self.training:
            with torch.no_grad():
                # Update usage counts
                hit_V = encodings.sum(dim=0)
                if self.record_hit == 0:
                    self.ema_vocab_hit.copy_(hit_V)
                elif self.record_hit < 100:
                    self.ema_vocab_hit.mul_(0.9).add_(hit_V * 0.1)
                else:
                    self.ema_vocab_hit.mul_(self._decay).add_(hit_V * (1 - self._decay))

                # Update embeddings
                dw = torch.matmul(encodings.t(), flat_input)
                self.ema_embedding.mul_(self._decay).add_(dw * (1 - self._decay))
                # Normalize by cluster size to update embeddings
                n = torch.sum(self.ema_vocab_hit)
                ema_cluster_size = (self.ema_vocab_hit + 1e-7) / (n + self._num_embeddings * 1e-7) * n
                self._embedding.weight.data.copy_(self.ema_embedding / (ema_cluster_size.unsqueeze(1) + 1e-7))

                self.record_hit += 1

        # Quantize inputs
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Compute losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Compute perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity


def cal_rec_loss(pred, target, mask, alpha, beta):

    rec_loss = mse_loss(pred, target, mask)

    # rec_loss = huber_loss(pred, target, mask)

    smo_loss = smooth_loss(pred, mode='dy1')

    return alpha * rec_loss + beta * smo_loss

def cal_nll_loss(mu, log_sigma, labels, mask):

    return nll_loss(mu, log_sigma, labels, mask)

def cal_nll_t(mu, log_sigma, df_raw, labels, mask):

    return nll_loss_student_t(mu, log_sigma, df_raw, labels, mask)




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.encoder = Encoder(in_channels=configs.enc_in, hidden_channels=64, embedding_dim=32, num_downsamples=4)
        # self.vq = VectorQuantizer(num_embeddings=512, embedding_dim=32, commitment_cost=0.25)
        self.vq = ResidualVectorQuantizer(num_embeddings=256, embedding_dim=32, commitment_cost=0.25, num_codebooks=8)
        # self.vq = VectorQuantizer2(
        #     vocab_size=2048, Cvae=32, using_znorm=False, beta=0.25,
        #     v_patch_nums=(3,6,12,23), quant_resi=0.5, share_quant_resi=4, eini=0.1, decay=0.99
        # )
        self.decoder = Decoder(out_channels=configs.enc_in, hidden_channels=64, embedding_dim=32, num_upsamples=4)
        self.transformer = imputation4missing(configs)
        self.scale = 16
        self.configs = configs
        print('mask ratio:', configs.mask_rate)

    def _prepare_input(self, x_enc, mask_ratio, valid_mask=None):
        """Prepare input tensor with padding and masking."""
        batch_size, seq_len, bands = x_enc.shape
        if valid_mask is None:
            valid_mask = (1 - torch.isnan(x_enc).int()).to('cuda')

        # Apply mask
        batch_x, batch_x_masked, missing_mask, indicating_mask = apply_mask(
            x_enc, valid_mask, mask_ratio, 'cuda'
        )

        # Pad to be divisible by self.scale
        original_steps = x_enc.shape[1]
        target_steps = ((original_steps + self.scale - 1) // self.scale) * self.scale
        masked_x_vae = F.pad(batch_x_masked, (0, 0, 0, target_steps - original_steps))
        masked_x_vae = masked_x_vae.permute(0, 2, 1)  # (batch, bands, steps)

        return batch_x, batch_x_masked, masked_x_vae, valid_mask, missing_mask, indicating_mask, original_steps

    def _apply_random_dropout(self, quantized, missing_mask, time_mark):
        """Apply random dropout for robustness."""
        if random.random() < 0.2:
            quantized = torch.zeros_like(quantized)
        if random.random() < 0.2:
            missing_mask = torch.zeros_like(missing_mask)
        if random.random() < 0.2:
            time_mark = torch.zeros_like(time_mark)
        return quantized, missing_mask, time_mark

    def _crop_output(self, output, log_sigma, original_steps):
        """Crop output to original sequence length."""
        x_recon = output[:, :original_steps, :]
        log_sigma = log_sigma[:, :original_steps, :]
        return x_recon, log_sigma

    def vae_train(self, x_enc, valid_mask=None):
        """Forward pass for VAE training."""
        batch_x, batch_x_masked, masked_x_vae, valid_mask, missing_mask, indicating_mask, original_steps = self._prepare_input(x_enc, 0,
                                                                                                        valid_mask)
        
        # Encoder and VQ-VAE
        z = self.encoder(masked_x_vae)
        z_batch, z_len, z_dim =  z.shape
        # dis_loss = dispersive_loss(z.view(z_batch, z_len*z_dim))
        vq_loss, quantized, perplexity = self.vq(z)

        # Decoder
        output, log_sigma, df = self.decoder(quantized)
        x_recon, log_sigma = self._crop_output(output, log_sigma, original_steps)

        # Loss calculation
        # loss = vq_loss + cal_rec_loss(x_recon, batch_x, valid_mask, 1, 0) + 0.2 * cal_nll_loss(x_recon, log_sigma, batch_x, valid_mask)
        loss = vq_loss + cal_rec_loss(x_recon, batch_x, valid_mask, 1, 0.2)  + 0.2 * nll_loss_student_t(x_recon, log_sigma, df, batch_x, valid_mask)
        return x_recon, loss, quantized, perplexity

    def transformer_train(self, x_enc, time_mark, valid_mask=None):
        """Forward pass for Transformer training."""
        batch_x, batch_x_masked, masked_x_vae, valid_mask, missing_mask, indicating_mask, original_steps = self._prepare_input(x_enc, self.configs.mask_rate,
                                                                                                        valid_mask)

        # VAE encoding
        z = self.encoder(masked_x_vae)
        _, quantized, _ = self.vq(z)
        quantized, missing_mask, time_mark = self._apply_random_dropout(quantized, missing_mask, time_mark)

        # Transformer
        trans_output = self.transformer(batch_x_masked, time_mark, missing_mask[:, :, 0].unsqueeze(-1).float(),
                                        quantized.detach())

        # Loss calculation
        loss = cal_rec_loss(trans_output, batch_x, indicating_mask, 1, 0.5)

        return trans_output, loss, trans_output, trans_output

    def vae_inference(self, x_enc, valid_mask=None):
        """Forward pass for VAE inference."""
        batch_x, _, masked_x_vae, valid_mask, _, _, original_steps = self._prepare_input(x_enc, self.configs.mask_rate, valid_mask)

        # Encoder and VQ-VAE
        z = self.encoder(masked_x_vae)
        _, quantized, _ = self.vq(z)

        # Decoder
        output, log_sigma, df = self.decoder(quantized)
        x_recon, log_sigma = self._crop_output(output, log_sigma, original_steps)
        # sigma = torch.exp(log_sigma) + 1e-6
        # sigma = sigma.sum(dim=-1, keepdim=True)
        nll = calculate_nll_t_per_timestep(x_recon, log_sigma, df, batch_x, valid_mask)
    
        anomalies = z_score_detector(nll[:,:,0], valid_mask[:,:,0])
     
        return x_recon, quantized, quantized, anomalies

    def transformer_inference(self, x_enc, time_mark, valid_mask=None):
        """Forward pass for Transformer inference."""
        batch_x, batch_x_masked, masked_x_vae, valid_mask, missing_mask, indicating_mask, original_steps = self._prepare_input(
            x_enc, self.configs.mask_rate,
            valid_mask)

        # VAE encoding
        z = self.encoder(masked_x_vae)
        _, quantized, _ = self.vq(z)

        # Transformer
        trans_output = self.transformer(batch_x_masked, time_mark, missing_mask[:, :, 0].unsqueeze(-1).float(),
                                        quantized)

        return trans_output

    def forward(self, x_enc, time_mark, valid_mask=None, mode='vae_train'):
        """Main forward method to dispatch based on mode."""
        if mode == 'vae_train':
            return self.vae_train(x_enc, valid_mask)
        elif mode == 'transformer_train':
            return self.transformer_train(x_enc, time_mark, valid_mask)
        elif mode == 'vae_inference':
            return self.vae_inference(x_enc, valid_mask)
        elif mode == 'transformer_inference':
            return self.transformer_inference(x_enc, time_mark, valid_mask)
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Choose from 'vae_train', 'transformer_train', 'vae_inference', 'transformer_inference'")