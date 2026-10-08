import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint
from layers.Embedding import PositionalEncoding
import torch.nn.functional as F


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class RotaryPositionEmbedding1D(nn.Module):
    def __init__(self, dim, frequency=10000, max_seq_len=1000):
        super().__init__()
        self.dim = dim
        self.frequency = frequency
        theta = 1.0 / (frequency ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('theta', theta)
        positions = torch.arange(max_seq_len)
        angles = positions.unsqueeze(-1) * theta
        angles = angles % (2 * math.pi)
        self.register_buffer('sin', torch.sin(angles))
        self.register_buffer('cos', torch.cos(angles))

    def forward(self, x, positions):
        if x.dim() == 4:
            batch, seq_len, num_heads, dim = x.shape
            assert dim == self.dim
            x = x.view(-1, seq_len, self.dim // 2, 2)
            sin = self.sin[:seq_len].to(x.device)
            cos = self.cos[:seq_len].to(x.device)
            x_rotated = torch.stack([
                x[..., 0] * cos - x[..., 1] * sin,
                x[..., 0] * sin + x[..., 1] * cos
            ], dim=-1).view(-1, seq_len, self.dim)
            x_rotated = x_rotated.view(batch, seq_len, num_heads, dim)
        else:
            seq_len = x.size(1)
            x = x.view(-1, seq_len, self.dim // 2, 2)
            sin = self.sin[:seq_len].to(x.device)
            cos = self.cos[:seq_len].to(x.device)
            x_rotated = torch.stack([
                x[..., 0] * cos - x[..., 1] * sin,
                x[..., 0] * sin + x[..., 1] * cos
            ], dim=-1).view(-1, seq_len, self.dim)
        return x_rotated


class PatchEmbedding(nn.Module):
    def __init__(
            self,
            seq_len: int = 100,
            season: int = 10,
            in_chans: int = 32,
            embed_dim: int = 64,  # 优化1：降低 dim
            norm_layer: callable = nn.LayerNorm
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.season = season
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.steps_per_season = math.ceil(seq_len / season)
        self.padding = self.steps_per_season * season - seq_len
        self.num_patches = self.season * self.steps_per_season
        self.proj = nn.Linear(in_chans, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert T == self.seq_len, f"Input sequence length {T} does not match expected {self.seq_len}"
        assert C == self.in_chans, f"Input channels {C} does not match expected {self.in_chans}"
        if self.padding > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, self.padding), mode='constant', value=0)
        x = x.view(B, self.season, self.steps_per_season, C)  # 优化2：使用 view 替代 rearrange
        x = self.proj(x)
        x = self.norm(x)
        return x

    def get_padding_mask(self) -> torch.Tensor:
        if self.padding == 0:
            return None
        mask = torch.ones(self.season, self.steps_per_season, device=self.proj.weight.device)
        if self.padding > 0:
            mask[:, -self.padding:] = 0
        return mask

class PatchEmbed(nn.Module):
    def __init__(
            self,
            seq_len: int = 100,
            kernel: int = 25,
            stride: int = 25,
            in_chans: int = 32,
            embed_dim: int = 64,
            norm_layer: callable = nn.LayerNorm
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.kernel = kernel
        self.stride = stride
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # Step 1: Calculate number of patches
        self.num_patches = math.ceil((seq_len - kernel + stride) / stride)
        if self.num_patches < 1:
            raise ValueError(f"Kernel size {kernel} is larger than sequence length {seq_len}")
        
        # Step 2: Calculate padding
        self.padded_seq_len = kernel + (self.num_patches - 1) * stride
        self.padding = self.padded_seq_len - seq_len
        
        # Step 3: Linear projection and normalization
        self.proj = nn.Linear(in_chans, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert T == self.seq_len, f"Input sequence length {T} does not match expected {self.seq_len}"
        assert C == self.in_chans, f"Input channels {C} does not match expected {self.in_chans}"
        
        # Step 4: Apply padding if needed
        if self.padding > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, self.padding), mode='constant', value=0)
        
        # Step 5: Extract patches using unfold
        x = x.unfold(dimension=1, size=self.kernel, step=self.stride)  # [B, num_patches, in_chans, kernel]
        x = x.transpose(2, 3)  # [B, num_patches, kernel, in_chans]
        
        # Step 6: Project patches
        x = self.proj(x)  # [B, num_patches, kernel, embed_dim]
        
        # Step 7: Normalize
        x = self.norm(x)
        
        return x

    def get_padding_mask(self) -> torch.Tensor:
        if self.padding == 0:
            return None
        mask = torch.ones(self.num_patches, self.kernel, device=self.proj.weight.device)
        if self.padding > 0:
            # Mask padded positions in the last patch
            last_patch_start = (self.num_patches - 1) * self.stride
            valid_steps = min(self.seq_len - last_patch_start, self.kernel)
            mask[-1, valid_steps:] = 0
        return mask

class TimesPatchEmbed(nn.Module):
    """
    统一的 Patching 和 Aggregation 模块：将 K * C_in 直接投影到 D_model。
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.kernel = configs.patch_len
        self.stride = configs.stride
        self.dropout = configs.dropout
        
        # C_in 是 (数据波段 + TimeMark + Mask) 的总维度
        self.c_in = configs.enc_in + 3
        
        # Step 1: 计算 Patch 数量 (N)
        self.num_patches = math.ceil((self.seq_len - self.kernel + self.stride) / self.stride)
        
        # Step 2: 计算 Padding
        self.padded_seq_len = self.kernel + (self.num_patches - 1) * self.stride
        self.padding = self.padded_seq_len - self.seq_len
        
        # 🎯 核心投影: K * C_in -> D_model (Tokenization + Aggregation)
        self.proj_unified = nn.Linear(self.kernel * self.c_in, self.d_model)
        
        # # Step 3: Positional Encoding (APE over sequence N)
        # self.position_encoding = PositionalEncoding(self.d_model, self.num_patches)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        # x_enc shape: [B, T, C_in]
        B, T, C = x_enc.shape

        # 1. Padding
        if self.padding > 0:
            # 填充在时间维度 T 的末尾
            x = torch.nn.functional.pad(x_enc, (0, 0, 0, self.padding), mode='constant', value=0)
        else:
            x = x_enc

        # 2. Extract Patches using unfold
        # x_unfold: [B, num_patches, kernel, C_in]
        x_unfold = x.unfold(dimension=1, size=self.kernel, step=self.stride)
        
        # 3. Flatten Patch and Channel dimensions
        # x_flat: [B, num_patches, kernel * C_in]
        x_flat = x_unfold.reshape(B, self.num_patches, self.kernel * self.c_in)
        
        # 4. 统一投影 (Aggregation + Tokenization)
        # x_embed: [B, num_patches, D_model]
        x_embed = self.proj_unified(x_flat) 
        
        # # 5. 添加位置编码 (APE over N)
        # x_with_pe = self.position_encoding(x_embed)
        
        return F.dropout(x_embed, p=self.dropout, training=self.training)

class IntraSeasonBlock(nn.Module):
    def __init__(self, dim, num_heads, num_register_tokens=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_register_tokens = num_register_tokens
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim)
        )
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.rope = RotaryPositionEmbedding1D(dim=self.head_dim, frequency=1000)

    def _forward(self, x, batch_size, season, steps_per_season, register_tokens, mask=None):
        device = x.device
        # Input shape: (batch_size, season, steps_per_season, dim)
        x = x.view(batch_size * season, steps_per_season, self.dim).contiguous()
        register_tokens = register_tokens.expand(batch_size * season, -1, -1)
        x = torch.cat([register_tokens, x],
                      dim=1)  # Shape: (batch_size * season, num_register_tokens + steps_per_season, dim)
        x_res = x

        q = self.query_projection(x)
        k = self.key_projection(x)
        v = self.value_projection(x)
        q = q.view(-1, self.num_register_tokens + steps_per_season, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_register_tokens + steps_per_season, self.num_heads, self.head_dim)

        positions = torch.arange(steps_per_season, device=device)
        positions = F.pad(positions, (self.num_register_tokens, 0), mode='constant', value=0)  # Pad at start
        q = self.rope(q, positions)
        k = self.rope(k, positions)
        q = q.view(-1, self.num_register_tokens + steps_per_season, self.dim)
        k = k.view(-1, self.num_register_tokens + steps_per_season, self.dim)

        if mask is not None:
            # Ensure mask is (batch_size * season, num_register_tokens + steps_per_season)
            mask = F.pad(mask, (self.num_register_tokens, 0), mode='constant',
                         value=False)  # False for register_tokens at start

        x, attn = self.attn(q, k, v, key_padding_mask=mask)
        x = x + x_res
        x = self.norm_attn(x)
        x = x[:, self.num_register_tokens:, :].contiguous().reshape(batch_size, season, steps_per_season, self.dim)
        x = x + self.ffn(x)
        x = self.norm_ffn(x)
        return x, attn

    def forward(self, x, batch_size, season, steps_per_season, register_tokens, mask=None):
        return checkpoint(self._forward, x, batch_size, season, steps_per_season, register_tokens, mask,
                          use_reentrant=False)


class InterSeasonBlock(nn.Module):
    def __init__(self, dim, num_heads, num_register_tokens=4, mlp_ratio=2, dropout=0.1, use_causal_attention=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_register_tokens = num_register_tokens
        self.use_causal_attention = use_causal_attention
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim)
        )
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.rope = RotaryPositionEmbedding1D(dim=self.head_dim, frequency=50)

    def _forward(self, x, batch_size, season, steps_per_season, register_tokens, mask=None):
        device = x.device
        # Input shape: (batch_size, season, steps_per_season, dim)
        x = x.reshape(batch_size * steps_per_season, season, self.dim).contiguous()
        register_tokens = register_tokens.expand(batch_size * steps_per_season, -1, -1)
        x = torch.cat([register_tokens, x],
                      dim=1)  # Shape: (batch_size * steps_per_season, num_register_tokens + season, dim)
        x_res = x

        q = self.query_projection(x)
        k = self.key_projection(x)
        v = self.value_projection(x)
        q = q.reshape(-1, season + self.num_register_tokens, self.num_heads, self.head_dim)
        k = k.reshape(-1, season + self.num_register_tokens, self.num_heads, self.head_dim)

        positions = torch.arange(season, device=device)
        positions = F.pad(positions, (self.num_register_tokens, 0), mode='constant',
                          value=0)  # Pad at start for register_tokens
        q = self.rope(q, positions)
        k = self.rope(k, positions)
        q = q.reshape(-1, season + self.num_register_tokens, self.dim)
        k = k.reshape(-1, season + self.num_register_tokens, self.dim)

        attn_mask = None
        if self.use_causal_attention:
            # Create causal mask for season + num_register_tokens
            sequence_length = season + self.num_register_tokens  # e.g., 14
            attn_mask = torch.triu(
                torch.ones(sequence_length, sequence_length, device=device), diagonal=1
            ).bool()  # Shape: (season + num_register_tokens, season + num_register_tokens), e.g., (14, 14)
            # Ensure register_tokens are fully visible (no masking)
            attn_mask[:, :self.num_register_tokens] = False  # register_tokens can attend to all
            attn_mask[:self.num_register_tokens, :] = False  # All can attend to register_tokens
            # Expand to (batch_size * steps_per_season * num_heads, sequence_length, sequence_length)
            attn_mask = attn_mask.unsqueeze(0).expand(batch_size * steps_per_season * self.num_heads, -1, -1)


        if mask is not None:
            # Ensure mask is (batch_size * steps_per_season, season + num_register_tokens)
            mask = F.pad(mask, (self.num_register_tokens, 0), mode='constant',
                         value=False)  # False for register_tokens at start

        x, attn = self.attn(q, k, v, attn_mask=attn_mask, key_padding_mask=mask)
        x = x + x_res
        x = self.norm_attn(x)
        x = x[:, self.num_register_tokens:, :].reshape(batch_size, season, steps_per_season,
                                                       self.dim)  # Extract season tokens
        x = x + self.ffn(x)
        x = self.norm_ffn(x)
        return x, attn

    def forward(self, x, batch_size, season, steps_per_season, register_tokens, mask=None):
        return checkpoint(self._forward, x, batch_size, season, steps_per_season, register_tokens, mask,
                         use_reentrant=False)


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=258, embedding_dim=32, commitment_cost=0.25, decay=0.99, eini=0.1,
                 num_codebooks=6, en_weight=0.1):
        super(ResidualVectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._num_codebooks = num_codebooks
        self._en_weight = en_weight

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
        entropy_loss = 0
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

            # Compute VQ losses
            e_latent_loss = F.mse_loss(quantized.detach(), residual)
            q_latent_loss = F.mse_loss(quantized, residual.detach())
            total_loss += q_latent_loss + self._commitment_cost * e_latent_loss

            # Compute entropy loss for this codebook
            counts = torch.sum(encodings, dim=0)  # Shape: (num_embeddings)
            probs = counts / (counts.sum() + 1e-6)  # Normalize to probabilities
            entropy = -torch.sum(probs * torch.log(probs + 1e-6))  # Negative entropy
            entropy_loss += entropy

            # Update residual
            residual = residual - quantized.detach()

            # EMA updates during training
            if self.training:
                with torch.no_grad():
                    hit_V = encodings.sum(dim=0)
                    if self.record_hit == 0:
                        self.ema_vocab_hit[i].copy_(hit_V)
                    elif self.record_hit < 100:
                        self.ema_vocab_hit[i].mul_(0.9).add_(hit_V * 0.1)
                    else:
                        self.ema_vocab_hit[i].mul_(self._decay).add_(hit_V * (1 - self._decay))

                    residual_plus_quantized = residual + quantized.detach()
                    residual_plus_quantized = torch.clamp(residual_plus_quantized, min=-100.0, max=100.0).float()
                    dw = torch.matmul(encodings.t(), residual_plus_quantized)
                    dw = torch.clamp(dw, min=-100.0, max=100.0)

                    self.ema_embedding[i] = self.ema_embedding[i].float()
                    self.ema_embedding[i].mul_(self._decay).add_(dw * (1 - self._decay))
                    self.ema_embedding[i] = torch.clamp(self.ema_embedding[i], min=-100.0, max=100.0)

                    n = torch.sum(self.ema_vocab_hit[i])
                    ema_cluster_size = (self.ema_vocab_hit[i] + 1e-5) / (n + self._num_embeddings * 1e-5) * n
                    ema_cluster_size = torch.clamp(ema_cluster_size, min=1e-5, max=1e7)

                    updated_embeddings = self.ema_embedding[i] / (ema_cluster_size.unsqueeze(1) + 1e-5)
                    updated_embeddings = torch.clamp(updated_embeddings, min=-100.0, max=100.0).half()
                    self._embeddings[i].weight.data.copy_(updated_embeddings)

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

        # Average entropy loss across codebooks
        entropy_loss /= self._num_codebooks

        return total_loss + self._en_weight * entropy_loss, quantized_all.permute(0, 2, 1).contiguous(), avg_perplexity, all_encodings

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.downsample = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.downsample(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, output_padding=0):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class DropPath(nn.Module):
    """Stochastic depth (per-sample residual drop), timm-style."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor.div_(keep_prob)
        return x * random_tensor


class AttentionBlock(nn.Module):
    """
    Standard Canonical Transformer Encoder Block (APE Mode)
    - Uses standard nn.MultiheadAttention (handles projection internally)
    - Removes custom RoPE and Register Token positional logic.
    - return_attn=False + need_weights=False 时 PyTorch 2.x 可走 SDPA/Flash，训练更快。
    - use_checkpoint=False 时省一次重算、加速明显，显存上升；默认 True 保持旧行为。
    - drop_path: 整层残差随机丢弃概率（stochastic depth），训练时按 batch 维随机。
    """
    def __init__(self, dim, num_heads, mlp_ratio=2, dropout=0.1, drop_path=0.0, use_checkpoint=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.drop_path = DropPath(drop_path)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim)
        )
        
        # 3. Layer Normalization
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

        
    def _forward(self, x, mask=None, is_causal=False, return_attn=False):
        # Input shape: (batch_size, steps, dim)
        seq_len = x.shape[1] # 获取序列长度
        shortcut = x

        attn_mask = None
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            attn_mask = torch.where(causal_mask.to(x.device), torch.tensor(-1e9).to(x.device), torch.tensor(0.0).to(x.device)) 

        x_attn, attn_weights = self.attn(
            query=x, 
            key=x, 
            value=x, 
            key_padding_mask=mask,
            attn_mask=attn_mask, 
            need_weights=return_attn,
        )
        if not return_attn:
            attn_weights = None

        x = x + x_attn # Residual Connection
        x = self.norm_attn(x)
        
        x_ffn_res = x # Residual for FFN layer
        x = self.ffn(x)
        
        x = x + x_ffn_res
        x = self.norm_ffn(x)

        # 单层一次随机深度：丢弃时本层输出回退为输入 shortcut
        x = shortcut + self.drop_path(x - shortcut)
        
        return x, attn_weights

    def forward(self, x, mask=None, is_causal=False, return_attn=False):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, mask, is_causal, return_attn, use_reentrant=False)
        return self._forward(x, mask, is_causal, return_attn)


class IntraSeason_AttnBlock(nn.Module):
    def __init__(self, dim, num_heads, num_register_tokens=4, mlp_ratio=2, rope_frequency=50, 
                 num_attn_blocks=2, dropout=0.1):
        super().__init__()
        
        # 堆叠多个 AttentionBlock
        self.num_attn_blocks = num_attn_blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(dim, num_heads, num_register_tokens, mlp_ratio, rope_frequency, dropout)
            for _ in range(num_attn_blocks)
        ])
    
    def _forward(self, x, mask=None):
        # x shape: (batch, season, steps_per_season, dim)
        batch, season, steps_per_season, dim = x.shape
        
        # 1. 重塑数据: (batch*season, steps_per_season, dim)
        x_reshaped = x.view(batch * season, steps_per_season, dim)
        # print('intra_attn x_reshaped.shape', x_reshaped.shape)
        # print('intra_attn mask.shape', mask.shape)
        # 2. 逐层 AttentionBlock（无 mask）
        attn_weights_list = []
        for attn_block in self.attention_blocks:
            x_reshaped, attn_weights = attn_block(x_reshaped, mask)
            attn_weights_list.append(attn_weights)
        
        # 3. 恢复原始形状: (batch, season, steps_per_season, dim)
        x_output = x_reshaped.view(batch, season, steps_per_season, dim)
        
        return x_output, attn_weights_list
    
    def forward(self, x, mask=None):
        return checkpoint(self._forward, x, mask, use_reentrant=False)

class InterSeason_AttnBlock(nn.Module):
    def __init__(self, dim, num_heads, num_register_tokens=4, mlp_ratio=2, rope_frequency=50, 
                 num_attn_blocks=2, dropout=0.1):
        super().__init__()
        
        # 堆叠多个 AttentionBlock
        self.num_attn_blocks = num_attn_blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(dim, num_heads, num_register_tokens, mlp_ratio, rope_frequency, dropout)
            for _ in range(num_attn_blocks)
        ])
    
    def _forward(self, x, mask=None):
        # x shape: (batch, season, steps_per_season, dim)
        batch, season, steps_per_season, dim = x.shape
        # 1. 重塑数据: (batch*season, steps_per_season, dim)
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(batch * steps_per_season, season, dim)
        
        # 2. 逐层 AttentionBlock（无 mask）
        attn_weights_list = []
        for attn_block in self.attention_blocks:
            x_reshaped, attn_weights = attn_block(x_reshaped, mask)
            attn_weights_list.append(attn_weights)
        
        # 3. 恢复原始形状: (batch, season, steps_per_season, dim)
        x_output = x_reshaped.view(batch, steps_per_season, season, dim).permute(0, 2, 1, 3)
        
        return x_output, attn_weights_list
    
    def forward(self, x, mask=None):
        return checkpoint(self._forward, x, mask, use_reentrant=False)     



# import torch
# import torch.nn as nn
# import math
# from torch.utils.checkpoint import checkpoint
# from layers.Embedding import PositionalEncoding
# import torch.nn.functional as F


# def make_2tuple(x):
#     if isinstance(x, tuple):
#         assert len(x) == 2
#         return x
#     assert isinstance(x, int)
#     return (x, x)


# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)


# # --------------- RoPE (DINOv3-style, 1D) ---------------
# def rope_rotate_half(x):
#     """RoPE rotate half: [x0,x1,x2,x3,...] -> [-x1,x0,-x3,x2,...] (DINOv3 attention.py)"""
#     x1, x2 = x.chunk(2, dim=-1)
#     return torch.cat([-x2, x1], dim=-1)


# def rope_apply(x, sin, cos):
#     """Apply RoPE: (x*cos) + (rotate_half(x)*sin) (DINOv3 attention.py)"""
#     return (x * cos) + (rope_rotate_half(x) * sin)


# class RopePositionEmbedding1D(nn.Module):
#     """
#     1D RoPE，对齐 DINOv3 的 RopePositionEmbedding 思路：无混合坐标、无可学习权重，
#     按 base 生成 periods，forward 时按当前序列长度动态计算 sin/cos，支持任意长度。
#     """
#     def __init__(self, head_dim: int, base: float = 10000.0, dtype=None, device=None):
#         super().__init__()
#         assert head_dim % 2 == 0
#         self.head_dim = head_dim
#         self.base = base
#         self.dtype = dtype or torch.get_default_dtype()
#         # periods[i] = base^(2i/head_dim), i=0..head_dim//2-1
#         periods = base ** (2 * torch.arange(head_dim // 2, dtype=torch.float64, device=device) / head_dim)
#         self.register_buffer("periods", periods.to(dtype=self.dtype), persistent=True)

#     def forward(self, seq_len: int, device=None) -> tuple:
#         """
#         返回 (sin, cos)，形状均为 [seq_len, head_dim]。
#         device 用于生成 sin/cos 所在设备；若为 None 则使用 self.periods.device。
#         """
#         device = device or self.periods.device
#         dtype = self.periods.dtype
#         periods = self.periods.to(device)
#         positions = torch.arange(seq_len, device=device, dtype=dtype)
#         angles = positions.unsqueeze(1) / periods.unsqueeze(0)  # [seq_len, head_dim//2]
#         sin = torch.sin(angles).repeat_interleave(2, dim=-1)   # [seq_len, head_dim]
#         cos = torch.cos(angles).repeat_interleave(2, dim=-1)   # [seq_len, head_dim]
#         return sin, cos


# class RotaryPositionEmbedding1D(nn.Module):
#     """旧版 1D RoPE（预计算 max_seq_len 的 sin/cos），保留兼容；新代码建议用 RopePositionEmbedding1D。"""
#     def __init__(self, dim, frequency=10000, max_seq_len=1000):
#         super().__init__()
#         self.dim = dim
#         self.frequency = frequency
#         theta = 1.0 / (frequency ** (torch.arange(0, dim, 2).float() / dim))
#         self.register_buffer('theta', theta)
#         positions = torch.arange(max_seq_len)
#         angles = positions.unsqueeze(-1) * theta
#         angles = angles % (2 * math.pi)
#         self.register_buffer('sin', torch.sin(angles))
#         self.register_buffer('cos', torch.cos(angles))

#     def forward(self, x, positions):
#         if x.dim() == 4:
#             batch, seq_len, num_heads, dim = x.shape
#             assert dim == self.dim
#             x = x.view(-1, seq_len, self.dim // 2, 2)
#             sin = self.sin[:seq_len].to(x.device)
#             cos = self.cos[:seq_len].to(x.device)
#             x_rotated = torch.stack([
#                 x[..., 0] * cos - x[..., 1] * sin,
#                 x[..., 0] * sin + x[..., 1] * cos
#             ], dim=-1).view(-1, seq_len, self.dim)
#             x_rotated = x_rotated.view(batch, seq_len, num_heads, dim)
#         else:
#             seq_len = x.size(1)
#             x = x.view(-1, seq_len, self.dim // 2, 2)
#             sin = self.sin[:seq_len].to(x.device)
#             cos = self.cos[:seq_len].to(x.device)
#             x_rotated = torch.stack([
#                 x[..., 0] * cos - x[..., 1] * sin,
#                 x[..., 0] * sin + x[..., 1] * cos
#             ], dim=-1).view(-1, seq_len, self.dim)
#         return x_rotated


# class PatchEmbedding(nn.Module):
#     def __init__(
#             self,
#             seq_len: int = 100,
#             season: int = 10,
#             in_chans: int = 32,
#             embed_dim: int = 64,  # 优化1：降低 dim
#             norm_layer: callable = nn.LayerNorm
#     ) -> None:
#         super().__init__()
#         self.seq_len = seq_len
#         self.season = season
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#         self.steps_per_season = math.ceil(seq_len / season)
#         self.padding = self.steps_per_season * season - seq_len
#         self.num_patches = self.season * self.steps_per_season
#         self.proj = nn.Linear(in_chans, embed_dim)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, T, C = x.shape
#         assert T == self.seq_len, f"Input sequence length {T} does not match expected {self.seq_len}"
#         assert C == self.in_chans, f"Input channels {C} does not match expected {self.in_chans}"
#         if self.padding > 0:
#             x = torch.nn.functional.pad(x, (0, 0, 0, self.padding), mode='constant', value=0)
#         x = x.view(B, self.season, self.steps_per_season, C)  # 优化2：使用 view 替代 rearrange
#         x = self.proj(x)
#         x = self.norm(x)
#         return x

#     def get_padding_mask(self) -> torch.Tensor:
#         if self.padding == 0:
#             return None
#         mask = torch.ones(self.season, self.steps_per_season, device=self.proj.weight.device)
#         if self.padding > 0:
#             mask[:, -self.padding:] = 0
#         return mask

# class PatchEmbed(nn.Module):
#     def __init__(
#             self,
#             seq_len: int = 100,
#             kernel: int = 25,
#             stride: int = 25,
#             in_chans: int = 32,
#             embed_dim: int = 64,
#             norm_layer: callable = nn.LayerNorm
#     ) -> None:
#         super().__init__()
#         self.seq_len = seq_len
#         self.kernel = kernel
#         self.stride = stride
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
        
#         # Step 1: Calculate number of patches
#         self.num_patches = math.ceil((seq_len - kernel + stride) / stride)
#         if self.num_patches < 1:
#             raise ValueError(f"Kernel size {kernel} is larger than sequence length {seq_len}")
        
#         # Step 2: Calculate padding
#         self.padded_seq_len = kernel + (self.num_patches - 1) * stride
#         self.padding = self.padded_seq_len - seq_len
        
#         # Step 3: Linear projection and normalization
#         self.proj = nn.Linear(in_chans, embed_dim)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, T, C = x.shape
#         assert T == self.seq_len, f"Input sequence length {T} does not match expected {self.seq_len}"
#         assert C == self.in_chans, f"Input channels {C} does not match expected {self.in_chans}"
        
#         # Step 4: Apply padding if needed
#         if self.padding > 0:
#             x = torch.nn.functional.pad(x, (0, 0, 0, self.padding), mode='constant', value=0)
        
#         # Step 5: Extract patches using unfold
#         x = x.unfold(dimension=1, size=self.kernel, step=self.stride)  # [B, num_patches, in_chans, kernel]
#         x = x.transpose(2, 3)  # [B, num_patches, kernel, in_chans]
        
#         # Step 6: Project patches
#         x = self.proj(x)  # [B, num_patches, kernel, embed_dim]
        
#         # Step 7: Normalize
#         x = self.norm(x)
        
#         return x

#     def get_padding_mask(self) -> torch.Tensor:
#         if self.padding == 0:
#             return None
#         mask = torch.ones(self.num_patches, self.kernel, device=self.proj.weight.device)
#         if self.padding > 0:
#             # Mask padded positions in the last patch
#             last_patch_start = (self.num_patches - 1) * self.stride
#             valid_steps = min(self.seq_len - last_patch_start, self.kernel)
#             mask[-1, valid_steps:] = 0
#         return mask

# class TimesPatchEmbed(nn.Module):
#     """
#     统一的 Patching 和 Aggregation 模块：将 K * C_in 直接投影到 D_model。
#     """
#     def __init__(self, configs):
#         super().__init__()
#         self.seq_len = configs.seq_len
#         self.d_model = configs.d_model
#         self.kernel = configs.patch_len
#         self.stride = configs.stride
#         self.dropout = configs.dropout
        
#         # C_in 是 (数据波段 + TimeMark + Mask) 的总维度
#         self.c_in = configs.enc_in + 3
        
#         # Step 1: 计算 Patch 数量 (N)
#         self.num_patches = math.ceil((self.seq_len - self.kernel + self.stride) / self.stride)
        
#         # Step 2: 计算 Padding
#         self.padded_seq_len = self.kernel + (self.num_patches - 1) * self.stride
#         self.padding = self.padded_seq_len - self.seq_len
        
#         # 🎯 核心投影: K * C_in -> D_model (Tokenization + Aggregation)
#         self.proj_unified = nn.Linear(self.kernel * self.c_in, self.d_model)
        
#         # # Step 3: Positional Encoding (APE over sequence N)
#         # self.position_encoding = PositionalEncoding(self.d_model, self.num_patches)

#     def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
#         # x_enc shape: [B, T, C_in]
#         B, T, C = x_enc.shape

#         # 1. Padding
#         if self.padding > 0:
#             # 填充在时间维度 T 的末尾
#             x = torch.nn.functional.pad(x_enc, (0, 0, 0, self.padding), mode='constant', value=0)
#         else:
#             x = x_enc

#         # 2. Extract Patches using unfold
#         # x_unfold: [B, num_patches, kernel, C_in]
#         x_unfold = x.unfold(dimension=1, size=self.kernel, step=self.stride)
        
#         # 3. Flatten Patch and Channel dimensions
#         # x_flat: [B, num_patches, kernel * C_in]
#         x_flat = x_unfold.reshape(B, self.num_patches, self.kernel * self.c_in)
        
#         # 4. 统一投影 (Aggregation + Tokenization)
#         # x_embed: [B, num_patches, D_model]
#         x_embed = self.proj_unified(x_flat) 
        
#         # # 5. 添加位置编码 (APE over N)
#         # x_with_pe = self.position_encoding(x_embed)
        
#         return F.dropout(x_embed, p=self.dropout, training=self.training)

# class IntraSeasonBlock(nn.Module):
#     def __init__(self, dim, num_heads, num_register_tokens=4, mlp_ratio=2, dropout=0.1):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.num_register_tokens = num_register_tokens
#         self.query_projection = nn.Linear(dim, dim)
#         self.key_projection = nn.Linear(dim, dim)
#         self.value_projection = nn.Linear(dim, dim)
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * mlp_ratio),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim * mlp_ratio, dim)
#         )
#         self.norm_attn = nn.LayerNorm(dim)
#         self.norm_ffn = nn.LayerNorm(dim)
#         self.rope = RotaryPositionEmbedding1D(dim=self.head_dim, frequency=1000)

#     def _forward(self, x, batch_size, season, steps_per_season, register_tokens, mask=None):
#         device = x.device
#         # Input shape: (batch_size, season, steps_per_season, dim)
#         x = x.view(batch_size * season, steps_per_season, self.dim).contiguous()
#         register_tokens = register_tokens.expand(batch_size * season, -1, -1)
#         x = torch.cat([register_tokens, x],
#                       dim=1)  # Shape: (batch_size * season, num_register_tokens + steps_per_season, dim)
#         x_res = x

#         q = self.query_projection(x)
#         k = self.key_projection(x)
#         v = self.value_projection(x)
#         q = q.view(-1, self.num_register_tokens + steps_per_season, self.num_heads, self.head_dim)
#         k = k.view(-1, self.num_register_tokens + steps_per_season, self.num_heads, self.head_dim)

#         positions = torch.arange(steps_per_season, device=device)
#         positions = F.pad(positions, (self.num_register_tokens, 0), mode='constant', value=0)  # Pad at start
#         q = self.rope(q, positions)
#         k = self.rope(k, positions)
#         q = q.view(-1, self.num_register_tokens + steps_per_season, self.dim)
#         k = k.view(-1, self.num_register_tokens + steps_per_season, self.dim)

#         if mask is not None:
#             # Ensure mask is (batch_size * season, num_register_tokens + steps_per_season)
#             mask = F.pad(mask, (self.num_register_tokens, 0), mode='constant',
#                          value=False)  # False for register_tokens at start

#         x, attn = self.attn(q, k, v, key_padding_mask=mask)
#         x = x + x_res
#         x = self.norm_attn(x)
#         x = x[:, self.num_register_tokens:, :].contiguous().reshape(batch_size, season, steps_per_season, self.dim)
#         x = x + self.ffn(x)
#         x = self.norm_ffn(x)
#         return x, attn

#     def forward(self, x, batch_size, season, steps_per_season, register_tokens, mask=None):
#         return checkpoint(self._forward, x, batch_size, season, steps_per_season, register_tokens, mask,
#                           use_reentrant=False)


# class InterSeasonBlock(nn.Module):
#     def __init__(self, dim, num_heads, num_register_tokens=4, mlp_ratio=2, dropout=0.1, use_causal_attention=False):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.num_register_tokens = num_register_tokens
#         self.use_causal_attention = use_causal_attention
#         self.query_projection = nn.Linear(dim, dim)
#         self.key_projection = nn.Linear(dim, dim)
#         self.value_projection = nn.Linear(dim, dim)
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * mlp_ratio),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim * mlp_ratio, dim)
#         )
#         self.norm_attn = nn.LayerNorm(dim)
#         self.norm_ffn = nn.LayerNorm(dim)
#         self.rope = RotaryPositionEmbedding1D(dim=self.head_dim, frequency=50)

#     def _forward(self, x, batch_size, season, steps_per_season, register_tokens, mask=None):
#         device = x.device
#         # Input shape: (batch_size, season, steps_per_season, dim)
#         x = x.reshape(batch_size * steps_per_season, season, self.dim).contiguous()
#         register_tokens = register_tokens.expand(batch_size * steps_per_season, -1, -1)
#         x = torch.cat([register_tokens, x],
#                       dim=1)  # Shape: (batch_size * steps_per_season, num_register_tokens + season, dim)
#         x_res = x

#         q = self.query_projection(x)
#         k = self.key_projection(x)
#         v = self.value_projection(x)
#         q = q.reshape(-1, season + self.num_register_tokens, self.num_heads, self.head_dim)
#         k = k.reshape(-1, season + self.num_register_tokens, self.num_heads, self.head_dim)

#         positions = torch.arange(season, device=device)
#         positions = F.pad(positions, (self.num_register_tokens, 0), mode='constant',
#                           value=0)  # Pad at start for register_tokens
#         q = self.rope(q, positions)
#         k = self.rope(k, positions)
#         q = q.reshape(-1, season + self.num_register_tokens, self.dim)
#         k = k.reshape(-1, season + self.num_register_tokens, self.dim)

#         attn_mask = None
#         if self.use_causal_attention:
#             # Create causal mask for season + num_register_tokens
#             sequence_length = season + self.num_register_tokens  # e.g., 14
#             attn_mask = torch.triu(
#                 torch.ones(sequence_length, sequence_length, device=device), diagonal=1
#             ).bool()  # Shape: (season + num_register_tokens, season + num_register_tokens), e.g., (14, 14)
#             # Ensure register_tokens are fully visible (no masking)
#             attn_mask[:, :self.num_register_tokens] = False  # register_tokens can attend to all
#             attn_mask[:self.num_register_tokens, :] = False  # All can attend to register_tokens
#             # Expand to (batch_size * steps_per_season * num_heads, sequence_length, sequence_length)
#             attn_mask = attn_mask.unsqueeze(0).expand(batch_size * steps_per_season * self.num_heads, -1, -1)


#         if mask is not None:
#             # Ensure mask is (batch_size * steps_per_season, season + num_register_tokens)
#             mask = F.pad(mask, (self.num_register_tokens, 0), mode='constant',
#                          value=False)  # False for register_tokens at start

#         x, attn = self.attn(q, k, v, attn_mask=attn_mask, key_padding_mask=mask)
#         x = x + x_res
#         x = self.norm_attn(x)
#         x = x[:, self.num_register_tokens:, :].reshape(batch_size, season, steps_per_season,
#                                                        self.dim)  # Extract season tokens
#         x = x + self.ffn(x)
#         x = self.norm_ffn(x)
#         return x, attn

#     def forward(self, x, batch_size, season, steps_per_season, register_tokens, mask=None):
#         return checkpoint(self._forward, x, batch_size, season, steps_per_season, register_tokens, mask,
#                          use_reentrant=False)


# class ResidualVectorQuantizer(nn.Module):
#     def __init__(self, num_embeddings=258, embedding_dim=32, commitment_cost=0.25, decay=0.99, eini=0.1,
#                  num_codebooks=6, en_weight=0.1):
#         super(ResidualVectorQuantizer, self).__init__()
#         self._embedding_dim = embedding_dim
#         self._num_embeddings = num_embeddings
#         self._commitment_cost = commitment_cost
#         self._decay = decay
#         self._num_codebooks = num_codebooks
#         self._en_weight = en_weight

#         # Create multiple codebooks
#         self._embeddings = nn.ModuleList([
#             nn.Embedding(self._num_embeddings, self._embedding_dim)
#             for _ in range(self._num_codebooks)
#         ])
#         # Initialize embedding weights for each codebook
#         for embedding in self._embeddings:
#             if eini > 0:
#                 nn.init.trunc_normal_(embedding.weight.data, std=eini)
#             else:
#                 embedding.weight.data.uniform_(-abs(eini) / self._num_embeddings, abs(eini) / self._num_embeddings)

#         # EMA buffers for each codebook
#         self.register_buffer('ema_vocab_hit', torch.zeros(self._num_codebooks, self._num_embeddings))
#         self.register_buffer('ema_embedding', torch.stack([emb.weight.data.clone() for emb in self._embeddings]))
#         self.register_buffer('record_hit', torch.tensor(0, dtype=torch.long))

#     def forward(self, inputs):
#         inputs = inputs.permute(0, 2, 1).contiguous()
#         input_shape = inputs.shape
#         residual = inputs.view(-1, self._embedding_dim)

#         total_loss = 0
#         entropy_loss = 0
#         quantized_all = torch.zeros_like(residual)
#         all_encodings = []
#         perplexities = []

#         # Iterate through each codebook
#         for i in range(self._num_codebooks):
#             # Compute distances to current codebook
#             distances = (torch.sum(residual ** 2, dim=1, keepdim=True) +
#                          torch.sum(self._embeddings[i].weight ** 2, dim=1) -
#                          2 * torch.matmul(residual, self._embeddings[i].weight.t()))
#             # Find nearest embeddings
#             encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#             encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
#             encodings.scatter_(1, encoding_indices, 1)
#             all_encodings.append(encoding_indices)

#             # Quantize for this codebook
#             quantized = torch.matmul(encodings, self._embeddings[i].weight)
#             quantized_all += quantized

#             # Compute VQ losses
#             e_latent_loss = F.mse_loss(quantized.detach(), residual)
#             q_latent_loss = F.mse_loss(quantized, residual.detach())
#             total_loss += q_latent_loss + self._commitment_cost * e_latent_loss

#             # Compute entropy loss for this codebook
#             counts = torch.sum(encodings, dim=0)  # Shape: (num_embeddings)
#             probs = counts / (counts.sum() + 1e-6)  # Normalize to probabilities
#             entropy = -torch.sum(probs * torch.log(probs + 1e-6))  # Negative entropy
#             entropy_loss += entropy

#             # Update residual
#             residual = residual - quantized.detach()

#             # EMA updates during training
#             if self.training:
#                 with torch.no_grad():
#                     hit_V = encodings.sum(dim=0)
#                     if self.record_hit == 0:
#                         self.ema_vocab_hit[i].copy_(hit_V)
#                     elif self.record_hit < 100:
#                         self.ema_vocab_hit[i].mul_(0.9).add_(hit_V * 0.1)
#                     else:
#                         self.ema_vocab_hit[i].mul_(self._decay).add_(hit_V * (1 - self._decay))

#                     residual_plus_quantized = residual + quantized.detach()
#                     residual_plus_quantized = torch.clamp(residual_plus_quantized, min=-100.0, max=100.0).float()
#                     dw = torch.matmul(encodings.t(), residual_plus_quantized)
#                     dw = torch.clamp(dw, min=-100.0, max=100.0)

#                     self.ema_embedding[i] = self.ema_embedding[i].float()
#                     self.ema_embedding[i].mul_(self._decay).add_(dw * (1 - self._decay))
#                     self.ema_embedding[i] = torch.clamp(self.ema_embedding[i], min=-100.0, max=100.0)

#                     n = torch.sum(self.ema_vocab_hit[i])
#                     ema_cluster_size = (self.ema_vocab_hit[i] + 1e-5) / (n + self._num_embeddings * 1e-5) * n
#                     ema_cluster_size = torch.clamp(ema_cluster_size, min=1e-5, max=1e7)

#                     updated_embeddings = self.ema_embedding[i] / (ema_cluster_size.unsqueeze(1) + 1e-5)
#                     updated_embeddings = torch.clamp(updated_embeddings, min=-100.0, max=100.0).half()
#                     self._embeddings[i].weight.data.copy_(updated_embeddings)

#             # Compute perplexity for this codebook
#             avg_probs = torch.mean(encodings, dim=0)
#             perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
#             perplexities.append(perplexity)

#         if self.training:
#             self.record_hit += 1

#         # Straight-through estimator
#         quantized_all = inputs.view(-1, self._embedding_dim) + (
#                 quantized_all - inputs.view(-1, self._embedding_dim)).detach()
#         quantized_all = quantized_all.view(input_shape)

#         # Average perplexity across codebooks
#         avg_perplexity = torch.mean(torch.stack(perplexities))

#         # Average entropy loss across codebooks
#         entropy_loss /= self._num_codebooks

#         return total_loss + self._en_weight * entropy_loss, quantized_all.permute(0, 2, 1).contiguous(), avg_perplexity, all_encodings

# class DownsampleBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
#         super().__init__()
#         self.downsample = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding
#         )
#         self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
#         self.activation = nn.SiLU()

#     def forward(self, x):
#         x = self.downsample(x)
#         x = self.norm(x)
#         x = self.activation(x)
#         return x

# class UpsampleBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, output_padding=0):
#         super().__init__()
#         self.upsample = nn.ConvTranspose1d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             output_padding=output_padding
#         )
#         self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
#         self.activation = nn.SiLU()

#     def forward(self, x):
#         x = self.upsample(x)
#         x = self.norm(x)
#         x = self.activation(x)
#         return x

# class AttentionBlock(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=2, dropout=0.1):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5

#         # Pre-Norm
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)

#         self.qkv = nn.Linear(dim, dim * 3)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
        
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * mlp_ratio),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim * mlp_ratio, dim),
#             nn.Dropout(dropout)
#         )
#         self.attn_drop = dropout

#     def _forward_compute(self, x, mask, is_causal, rope, return_attn):
#         """
#         核心计算逻辑：Pre-Norm + RoPE + SDPA
#         """
#         B, L, C = x.shape
#         shortcut = x
        
#         # 1. Attention 路径 (Pre-Norm)
#         x = self.norm1(x)
#         qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         # RoPE 应用
#         if rope is not None:
#             sin, cos = rope
#             prefix = L - sin.shape[0]
#             if prefix > 0:
#                 q[:, :, prefix:] = (q[:, :, prefix:] * cos) + (rope_rotate_half(q[:, :, prefix:]) * sin)
#                 k[:, :, prefix:] = (k[:, :, prefix:] * cos) + (rope_rotate_half(k[:, :, prefix:]) * sin)
#             else:
#                 q = (q * cos) + (rope_rotate_half(q) * sin)
#                 k = (k * cos) + (rope_rotate_half(k) * sin)

#         # 构造掩码
#         attn_mask = None
#         if is_causal or (mask is not None):
#             if is_causal:
#                 attn_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
#             if mask is not None:
#                 p_mask = mask.view(B, 1, 1, L).expand(-1, 1, L, -1).to(torch.bool)
#                 attn_mask = (attn_mask | p_mask) if attn_mask is not None else p_mask

#         # 计算权重或直接 SDPA
#         if not return_attn:
#             x_attn = F.scaled_dot_product_attention(
#                 q, k, v, attn_mask=attn_mask,
#                 dropout_p=self.attn_drop if self.training else 0.0,
#                 scale=self.scale
#             )
#             attn_weights = None
#         else:
#             scores = (q @ k.transpose(-2, -1)) * self.scale
#             if attn_mask is not None:
#                 scores.masked_fill_(attn_mask, float("-inf"))
#             attn_weights = F.softmax(scores, dim=-1)
#             x_attn = attn_weights @ v

#         x_attn = x_attn.transpose(1, 2).reshape(B, L, C)
#         x = shortcut + self.proj_drop(self.proj(x_attn))

#         # 2. MLP 路径 (Pre-Norm)
#         x = x + self.ffn(self.norm2(x))
        
#         return x, attn_weights

#     def forward(self, x, mask=None, is_causal=False, rope=None, return_attn=False):
#         # 只有在训练模式且非可视化模式下使用 checkpoint，以节省显存
#         if self.training and not return_attn:
#             # 注意：PyTorch 默认要求传入 checkpoint 的函数返回值中不能有 None，
#             # 这里的 _forward_compute 会返回 (x, None)，我们需要处理一下。
#             # 或者使用 use_reentrant=False (推荐)
#             return torch.utils.checkpoint.checkpoint(
#                 self._forward_compute, x, mask, is_causal, rope, return_attn,
#                 use_reentrant=False
#             )
#         else:
#             return self._forward_compute(x, mask, is_causal, rope, return_attn)


# class IntraSeason_AttnBlock(nn.Module):
#     def __init__(self, dim, num_heads, num_register_tokens=4, mlp_ratio=2, rope_frequency=50, 
#                  num_attn_blocks=2, dropout=0.1):
#         super().__init__()
        
#         # 堆叠多个 AttentionBlock
#         self.num_attn_blocks = num_attn_blocks
#         self.attention_blocks = nn.ModuleList([
#             AttentionBlock(dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
#             for _ in range(num_attn_blocks)
#         ])
    
#     def _forward(self, x, mask=None):
#         # x shape: (batch, season, steps_per_season, dim)
#         batch, season, steps_per_season, dim = x.shape
        
#         # 1. 重塑数据: (batch*season, steps_per_season, dim)
#         x_reshaped = x.view(batch * season, steps_per_season, dim)
#         # print('intra_attn x_reshaped.shape', x_reshaped.shape)
#         # print('intra_attn mask.shape', mask.shape)
#         # 2. 逐层 AttentionBlock（无 mask）
#         attn_weights_list = []
#         for attn_block in self.attention_blocks:
#             x_reshaped, attn_weights = attn_block(x_reshaped, mask)
#             attn_weights_list.append(attn_weights)
        
#         # 3. 恢复原始形状: (batch, season, steps_per_season, dim)
#         x_output = x_reshaped.view(batch, season, steps_per_season, dim)
        
#         return x_output, attn_weights_list
    
#     def forward(self, x, mask=None):
#         return checkpoint(self._forward, x, mask, use_reentrant=False)

# class InterSeason_AttnBlock(nn.Module):
#     def __init__(self, dim, num_heads, num_register_tokens=4, mlp_ratio=2, rope_frequency=50, 
#                  num_attn_blocks=2, dropout=0.1):
#         super().__init__()
        
#         # 堆叠多个 AttentionBlock
#         self.num_attn_blocks = num_attn_blocks
#         self.attention_blocks = nn.ModuleList([
#             AttentionBlock(dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
#             for _ in range(num_attn_blocks)
#         ])
    
#     def _forward(self, x, mask=None):
#         # x shape: (batch, season, steps_per_season, dim)
#         batch, season, steps_per_season, dim = x.shape
#         # 1. 重塑数据: (batch*season, steps_per_season, dim)
#         x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(batch * steps_per_season, season, dim)
        
#         # 2. 逐层 AttentionBlock（无 mask）
#         attn_weights_list = []
#         for attn_block in self.attention_blocks:
#             x_reshaped, attn_weights = attn_block(x_reshaped, mask)
#             attn_weights_list.append(attn_weights)
        
#         # 3. 恢复原始形状: (batch, season, steps_per_season, dim)
#         x_output = x_reshaped.view(batch, steps_per_season, season, dim).permute(0, 2, 1, 3)
        
#         return x_output, attn_weights_list
    
#     def forward(self, x, mask=None):
#         return checkpoint(self._forward, x, mask, use_reentrant=False)     


