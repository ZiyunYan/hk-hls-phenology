import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import AdaLN_EncoderLayer, AdaLN_Encoder, EncoderLayer
from layers.Transformer_EncDec import Encoder as Transformer_Encoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.SaitsEmbedding import SaitsEmbedding
from utils.tools import apply_mask
import sys

class latent_transfrmer(nn.Module):
    def __init__(self, configs):
        super(latent_transfrmer, self).__init__()
        # self.d_model = configs.d_model // 2

        self.embedding = SaitsEmbedding(
            d_in=configs.d_model // 2 + 3,
            d_model=configs.d_model,
            with_pos=True,
            embed_type=configs.embed,
            freq=configs.freq,
            n_max_steps=12,
            dropout=0.1
        )

        self.time_embed = nn.Linear(366, 12)
        self.mask_embed = nn.Linear(366, 12)

        # Encoder with AdaLN
        self.encoder = Transformer_Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, diag_mask_flag=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.final_mlp = nn.Linear(configs.d_model, configs.d_model//2)

    def forward(self, x_enc, missing_mask, time_mark):
        # print('missing', missing_mask, missing_mask.shape)
        # print('time', time_mark, time_mark.shape)
        mask_embed = self.mask_embed(missing_mask[:,:,0].unsqueeze(-1).permute(0,2,1)).permute(0,2,1)
        time_embed = self.time_embed(time_mark.permute(0, 2, 1)).permute(0, 2, 1)
        enc_in = self.embedding(x_enc.permute(0,2,1), mask_embed, time_embed)
        enc_out, attns = self.encoder(enc_in)
        enc_out = self.final_mlp(enc_out)
        return enc_out.permute(0,2,1)


class imputation4missing(nn.Module):
    def __init__(self, configs):
        super(imputation4missing, self).__init__()
        self.pred_len = configs.label_len
        self.output_attention = True
        self.d_model = configs.d_model


        # 嵌入层
        self.embedding = SaitsEmbedding(
            d_in=configs.enc_in + 3,
            d_model=configs.d_model,
            with_pos=True,
            embed_type=configs.embed,
            freq=configs.freq,
            n_max_steps=self.pred_len,
            dropout=0.1
        )

        self.coondition_mlp = nn.Linear(64*12, configs.d_model)
        # VQVAE 码表嵌入
        self.codebook_embed = nn.Embedding(64, configs.d_model)

        # # 条件嵌入（时间标记和掩码）
        # self.time_embed = nn.Linear(2, configs.d_model)
        # self.mask_embed = nn.Linear(1, configs.d_model)

        # Encoder with AdaLN
        self.encoder = AdaLN_Encoder(
            [
                AdaLN_EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, diag_mask_flag=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
        )

        self.output_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, missing_mask, vq):
        # x_enc: [batch, seq_len, enc_in]
        # x_mark_enc: [batch, seq_len, 3]
        # missing_mask: [batch, seq_len, 1]
        # codebook_indices: [batch, seq_len]
        batch, seq, bands = x_enc.shape
        # 嵌入输入
        enc_out = self.embedding(x_enc, missing_mask[:,:,0].unsqueeze(-1), x_mark_enc)  # [batch, seq_len, d_model]

        # time_embed = self.time_embed(x_mark_enc)  # [batch, seq_len, d_model]
        # mask_embed = self.mask_embed(missing_mask)  # [batch, seq_len, d_model]

        # condition = codebook_embed + time_embed + mask_embed  # [batch, seq_len, d_model]
        condition = self.coondition_mlp(vq.view(batch, -1))
        # print('condition shape', condition.shape)
        # Encoder with AdaLN
        enc_out, attns = self.encoder(enc_out, condition=condition)

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
        self.final_residual = ResidualBlock(current_channels, current_channels)
        self.final_residual1 = ResidualBlock(current_channels, current_channels)
        self.atten = TemporalNonLocalBlock(current_channels)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=current_channels)
        # 投影到嵌入空间
        self.pre_vq_conv = nn.Conv1d(
            in_channels=current_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1
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
        x = self.final_residual(x)
        # print('enc final_residual', x.shape)
        x = self.atten(x)
        x = self.final_residual1(x)
        # print('enc final_residual1', x.shape)
        x = self.norm(x)
        x = self.swish(x)
        # 投影到嵌入空间
        z = self.pre_vq_conv(x)

        return z

class Decoder(nn.Module):
    def __init__(self, out_channels=4, hidden_channels=64, embedding_dim=64, num_upsamples=4):
        super(Decoder, self).__init__()

        # 计算起始通道数（适配编码器的通道数变化）
        self.init_channels = hidden_channels * (2 ** (num_upsamples // 2))

        # 初始卷积层
        self.init_conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=self.init_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 初始残差块
        self.init_residual = ResidualBlock(self.init_channels, self.init_channels)

        # 构建解码层
        self.layers = nn.ModuleList()
        current_channels = self.init_channels

        for i in range(num_upsamples):
            # Only halve channels every two upsampling steps (i.e., when i is odd: 1, 3, ...)
            if i % 2 == 1:
                out_channels_block = current_channels // 2
            else:
                out_channels_block = current_channels  # Keep channels the same

            self.layers.append(nn.ModuleList([
                ResidualBlock(current_channels, current_channels),  # Keep channel count
                UpSampleBlock(current_channels),  # Upsample time dimension by 2x
                ResidualBlock(current_channels, out_channels_block)  # Change channels (same or halve)
            ]))

            current_channels = out_channels_block

        # 最终输出层
        self.final_residual = ResidualBlock(current_channels, current_channels)
        self.final_conv = nn.Conv1d(
            in_channels=current_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, z):
        # 初始特征转换
        x = self.init_conv(z)
        # print('dec conv1', x.shape)
        x = self.init_residual(x)
        # print('dec res1', x.shape)
        # 通过解码层
        for residual1, upsample, residual2 in self.layers:
            x = residual1(x)  # Keep channel count
            x = upsample(x)  # Time dimension upsample (2x)
            x = residual2(x)  # Change channel count (same or halve)
        # print('de conv', x.shape)
        # 最终特征处理和输出
        x = self.final_residual(x)
        # print('de final res', x.shape)
        x = self.final_conv(x)

        return x

# VectorQuantizer 模块（保持不变）
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=258, embedding_dim=64, commitment_cost=0.25, decay=0.99, eini=0.1):
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
        print('inputs', inputs.shape)
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        print('flat inputs', flat_input.shape)

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


# VQ-VAE 模型
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.encoder = Encoder(configs.enc_in, hidden_channels=96, embedding_dim=64, num_downsamples=5)
        self.vq = VectorQuantizer(num_embeddings=4096, embedding_dim=32, commitment_cost=0.25)
        self.decoder = Decoder(configs.enc_in, hidden_channels=96, embedding_dim=64, num_upsamples=5)
        self.transformer = imputation4missing(configs)
        self.latent_transformer = latent_transfrmer(configs)
        self.scale = 32
        self.configs = configs

    def forward(self, x_enc, time_mark, valid_mask=None, ano_mask=None):


        # batch_size, seq_len, bands = x_enc.shape
        # if valid_mask is None:
        #     valid_mask  = (1 - torch.isnan(x_enc).int()).to('cuda')

        # # 0 mask ratio for vqvae
        # batch_x, batch_x_masked, missing_mask, indicating_mask = apply_mask(
        #     x_enc, valid_mask, self.configs.mask_rate, 'cuda'
        # )
        # 输入：(batch, steps, bands) -> (batch, bands, steps)
        original_steps = x_enc.shape[1]
        # 填充到能被 self.scale 整除的长度
        target_steps = ((original_steps + self.scale - 1) // self.scale) * self.scale
        x = F.pad(x_enc, (0, 0, 0, target_steps - original_steps))  # 在 steps 维度填充
        x = x.permute(0, 2, 1)  # (batch, bands, steps)

        z = self.encoder(x)

        vq_loss, quantized, perplexity = self.vq(z)

        print('vq', quantized.shape)
        # dec_out = self.transformer2(batch_x_masked, time_mark, missing_mask, quantized)
        # quantized = self.latent_transformer(quantized, missing_mask.float(), time_mark)
        x_recon = self.decoder(quantized)

        # 裁剪回原始长度
        x_recon = x_recon[:, :, :original_steps]
        x_recon = x_recon.permute(0, 2, 1)  # (batch, steps, bands)
        # print(perplexity)
        sys.exit(1)
        return x_recon, vq_loss, quantized, x_recon