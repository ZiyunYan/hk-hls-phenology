import torch
import torch.nn as nn
import torch.nn.functional as F
from models.quant import VectorQuantizer2
__all__ = ['Encoder', 'Decoder']


# 激活函数
def nonlinearity(x):
    return x * torch.sigmoid(x)


# 归一化层
def Normalize(in_channels):
    return torch.nn.InstanceNorm1d(num_features=in_channels, eps=1e-6, affine=True)


# 1D下采样
class Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1), mode='constant', value=0))


# 1D上采样
class Upsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


# 1D残差块
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h


# 1D注意力块
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels

        self.norm = Normalize(in_channels)
        self.qkv = torch.nn.Conv1d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        qkv = self.qkv(self.norm(x))
        B, _, T = qkv.shape  # B, 3C, T
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, T).unbind(1)

        # 计算注意力
        q = q.permute(0, 2, 1).contiguous()  # B, T, C
        k = k.permute(0, 2, 1).contiguous()  # B, T, C
        w = torch.bmm(q, k.transpose(1, 2)).mul_(self.w_ratio)  # B, T, T
        w = F.softmax(w, dim=2)

        # 对值进行注意力加权
        v = v.permute(0, 2, 1).contiguous()  # B, T, C
        h = torch.bmm(w, v)  # B, T, C
        h = h.permute(0, 2, 1).contiguous()  # B, C, T

        return x + self.proj_out(h)


# # 创建注意力模块
# def make_attn(in_channels, using_sa=True):
#     return AttnBlock(in_channels) if using_sa else nn.Identity()

def make_attn(in_channels, using_sa=True):
    if not using_sa:
        return nn.Identity()  # Return identity if attention is disabled
    return TemporalSelfAttention(in_channels)


class TemporalSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=1):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        # Ensure in_channels is divisible by num_heads
        if in_channels % num_heads != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})")
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,  # Channels as embedding dimension
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1  # Optional dropout for regularization
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        # x: (batch, channels, steps)
        batch, channels, steps = x.shape

        # Transpose to (batch, steps, channels) for attention on time steps
        x = x.transpose(1, 2)  # (batch, steps, channels)

        # Apply multi-head self-attention
        attn_output, _ = self.attention(x, x, x)

        # Residual connection and layer normalization
        x = self.norm(x + attn_output)

        # Transpose back to (batch, channels, steps)
        x = x.transpose(1, 2)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=32, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) +
                     torch.sum(self._embedding.weight ** 2, dim=1) -
                     2 * torch.matmul(flat_input, self._embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity

class Decoder(nn.Module):
    def __init__(self, z_channels: int, input_features: int, T: int, timesteps: int):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(z_channels, z_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv1d(z_channels // 2, input_features, kernel_size=3, stride=1, padding=1)
        self.upsample_factor = timesteps // T

    def forward(self, f_hat: torch.Tensor) -> torch.Tensor:
        x = self.upsample(f_hat)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=self.upsample_factor // 2, mode='linear')
        x = self.conv(x)
        return x

class TimeSeriesDiscriminator(nn.Module):
    def __init__(self, bands: int, num_filters_last: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv1d(bands, num_filters_last, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv1d(
                    num_filters_last * num_filters_mult_last,
                    num_filters_last * num_filters_mult,
                    kernel_size=5,
                    stride=2 if i < n_layers else 1,
                    padding=2,
                    bias=False
                ),
                nn.BatchNorm1d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        layers.append(nn.Conv1d(num_filters_last * num_filters_mult, 1, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [batch_size, bands, steps]
        output = self.model(x)  # [batch_size, 1, steps / 8]
        return output

# 编码器
class Encoder(nn.Module):
    def __init__(
            self, *, ch=64, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
            dropout=0.0, in_channels=3,
            z_channels, double_z=False, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # 输入卷积
        self.conv_in = torch.nn.Conv1d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)

        # 中间层
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # 输出层
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1,
                                        padding=1)

    def forward(self, x):
        # x: (batch, steps, bands)
        x = x.transpose(1, 2)  # (batch, bands, steps)
        # 下采样
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # 中间层
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))

        # 输出
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h


# 解码器
class Decoder(nn.Module):
    def __init__(
            self, *, ch=64, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
            dropout=0.0, in_channels=3,
            z_channels, double_z=False, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # 计算最低分辨率的输入通道
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # 输入卷积
        self.conv_in = torch.nn.Conv1d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # 中间层
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # 上采样
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample2x(block_in)
            self.up.insert(0, up)

        # 输出层
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # z: (batch, z_channels, steps)
        # 中间层
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))

        # 上采样
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # 输出
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h.transpose(1, 2)  # (batch, steps, bands)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 手动设置 Encoder 参数
        encoder_configs = {
            'ch': 32,
            'ch_mult': (1, 2, 4, 8),  # 8 倍下采样
            'num_res_blocks': 2,
            'dropout': 0.0,
            'in_channels': 7,  # bands=4 + missing_mask=1 + time_mark=2
            'z_channels': 32,
            'double_z': False,
            'using_sa': True,
            'using_mid_sa': True,
        }
        decoder_configs = {
            'ch': 32,
            'ch_mult': (1, 2, 4, 8),  # 8 倍下采样
            'num_res_blocks': 2,
            'dropout': 0.0,
            'in_channels': 4,  # bands=4
            'z_channels': 32,
            'double_z': False,
            'using_sa': True,
            'using_mid_sa': True,
        }
        self.encoder = Encoder(**encoder_configs)
        self.decoder = Decoder(**decoder_configs)
        self.v_patch_nums = (1, 2, 4, 8, 16, 32, 368//8)  # 时间分辨率

        self.vq = VectorQuantizer2(
            vocab_size=512,
            Cvae=32,
            using_znorm=False,
            v_patch_nums=self.v_patch_nums,
            share_quant_resi=4
        )
        # self.vq = VectorQuantizer()

        self.downsample_ratio = self.encoder.downsample_ratio  # 8

    def forward(self, x, time_mark, missing_mask, ano_mask=None):
        # x: (batchsize, steps, bands)
        # time_mark: (batchsize, steps, 2)
        # missing_mask: (batchsize, steps, bands)
        # ano_mask: (batchsize, steps, bands) or None

        # 检查时间步是否需要填充
        batchsize, steps, bands = x.shape
        if steps % self.downsample_ratio != 0:
            # 计算需要填充的步数
            padded_steps = ((steps + self.downsample_ratio - 1) // self.downsample_ratio) * self.downsample_ratio
            pad_size = padded_steps - steps

            # 填充 x（填充 0）
            x = F.pad(x, (0, 0, 0, pad_size), mode='constant', value=0)  # (batchsize, padded_steps, bands)

            # 填充 time_mark（填充 0）
            time_mark = F.pad(time_mark, (0, 0, 0, pad_size), mode='constant', value=0)  # (batchsize, padded_steps, 2)

            # 填充 missing_mask（填充 0，表示缺失）
            missing_mask = F.pad(missing_mask, (0, 0, 0, pad_size), mode='constant',
                                 value=0)  # (batchsize, padded_steps, bands)

            # 填充 ano_mask（如果存在，填充 0）
            if ano_mask is not None:
                ano_mask = F.pad(ano_mask, (0, 0, 0, pad_size), mode='constant',
                                 value=0)  # (batchsize, padded_steps, bands)

        # 处理缺失值
        # x = x * missing_mask

        # 拼接 time_mark
        x = torch.cat([x, missing_mask[:,:,0].unsqueeze(-1), time_mark], dim=-1)  # (batchsize, padded_steps, bands + 2)

        # 编码
        z = self.encoder(x)  # (batchsize, z_channels, padded_steps/8)
        f_hat, usages, mean_vq_ossl = self.vq(z, ret_usages=True)
        # mean_vq_ossl, f_hat, perplexity = self.vq(z)
        x_out = self.decoder(f_hat)
        # 裁剪输出（恢复原始 steps）
        x_out = x_out[:, :steps, :]  # (batchsize, steps, bands + 2)
        # missing_mask = missing_mask[:, :steps, :]  # (batchsize, steps, bands)
        # if ano_mask is not None:
        #     ano_mask = ano_mask[:, :steps, :]  # (batchsize, steps, bands)

        return x_out, mean_vq_ossl, x_out, None