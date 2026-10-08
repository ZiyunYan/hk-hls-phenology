import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # 定义检查点内部的计算逻辑
        def checkpoint_fn(x, attn_mask, tau, delta):
            new_x, attn = self.attention(
                x, x, x,
                attn_mask=attn_mask,
                tau=tau, delta=delta
            )
            x = x + self.dropout(new_x)
            y = self.norm1(x)
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            return self.norm2(x + y), attn

        # 使用 checkpoint 包装 forward 计算
        if self.training:
            x, attn = checkpoint(checkpoint_fn, x, attn_mask, tau, delta, use_reentrant=False)
        else:
            x, attn = checkpoint_fn(x, attn_mask, tau, delta)

        return x, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta_i = delta if i == 0 else None
                # 使用 checkpoint 包装 attn_layer 的 forward
                if self.training:
                    x, attn = checkpoint(attn_layer, x, attn_mask, tau, delta_i, use_reentrant=False)
                else:
                    x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta_i)
                x = conv_layer(x)
                attns.append(attn)
            # 最后一层注意力层
            if self.training:
                x, attn = checkpoint(self.attn_layers[-1], x, attn_mask, tau, None, use_reentrant=False)
            else:
                x, attn = self.attn_layers[-1](x, attn_mask=attn_mask, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                # 使用 checkpoint 包装 attn_layer 的 forward
                if self.training:
                    x, attn = checkpoint(attn_layer, x, attn_mask, tau, delta, use_reentrant=False)
                else:
                    x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):

        new_x, attns = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attns


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, attns


# AdaptiveLayerNorm（带MLP）
class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2)  # 输出 scale 和 shift
        )

        # 零初始化 MLP 的最后一层权重和偏置
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, condition):
        # x: [batch, seq_len, d_model]
        # condition: [batch, d_model]
        norm_x = self.norm(x)  # [batch, seq_len, d_model]

        # 通过 MLP 生成 scale 和 shift
        scale_shift = self.mlp(condition)  # [batch, d_model * 2]
        scale_shift = scale_shift.unsqueeze(1)  # [batch, 1, d_model * 2]
        scale, shift = scale_shift.chunk(2, dim=-1)  # 各 [batch, 1, d_model]

        # 应用 scale 和 shift
        return norm_x * (1 + scale) + shift  # [batch, seq_len, d_model]

# AdaLN_EncoderLayer
class AdaLN_EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(AdaLN_EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = AdaptiveLayerNorm(d_model)  # 替换为 AdaLN
        self.norm2 = AdaptiveLayerNorm(d_model)  # 替换为 AdaLN
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, condition=None, attn_mask=None, tau=None, delta=None):
        # x: [batch, seq_len, d_model]
        # condition: [batch, seq_len, d_model]
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x, condition)  # 使用 AdaLN
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y, condition), attn  # 使用 AdaLN

# AdaLN_Encoder
class AdaLN_Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=AdaptiveLayerNorm(128), conv_layers=None):
        super(AdaLN_Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer  # 应为 AdaptiveLayerNorm 或 None

    def forward(self, x, condition=None, attn_mask=None, tau=None, delta=None):
        # x: [batch, seq_len, d_model]
        # condition: [batch, seq_len, d_model]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, condition=condition, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, condition=condition, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, condition=condition, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x, condition)

        return x, attns

def modulate(x, shift, scale):
    """
    自适应层归一化 (adaLN) 的调制函数
    x: 归一化后的输入张量，形状 (N, T, D)
    shift: 偏移参数，形状 (N, T, D) 或 (N, D)
    scale: 缩放参数，形状 (N, T, D) 或 (N, D)
    返回: 调制后的张量，形状 (N, T, D)
    """
    if shift.dim() == 2:  # 全局条件，广播到 T
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift

def init_adaLN_zero(m):
    """
    初始化 adaLN-Zero，确保 gate 参数为 0，scale 参数接近 0，shift 参数为 0
    """
    if isinstance(m, nn.Linear) and m.out_features == 6 * m.in_features:
        d_model = m.in_features
        nn.init.zeros_(m.weight[-2 * d_model:])
        nn.init.zeros_(m.bias[-2 * d_model:])
        nn.init.constant_(m.weight[:2 * d_model], 0.1)
        nn.init.zeros_(m.bias[:2 * d_model])
        nn.init.zeros_(m.weight[2 * d_model:4 * d_model])
        nn.init.zeros_(m.bias[2 * d_model:4 * d_model])

class DiTEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, num_steps, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # adaLN 调制模块，接受序列条件 (N, T, D)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )
        self.num_steps = num_steps

    def forward(self, x, c, attn_mask=None, tau=None, delta=None):
        """
        x: 输入张量，形状 (N, T, D)
        c: 序列条件，形状 (N, T, D)
        attn_mask: 注意力掩码（可选）
        tau, delta: 注意力模块的额外参数（可选）
        返回: (更新后的 x, 注意力输出)
        """
        # c: (N, T, D) -> (N, T, 6 * D)
        mod_params = self.adaLN_modulation(c).view(-1, self.num_steps, 6, self.adaLN_modulation[-1].out_features // 6)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_params.split(1, dim=2)
        shift_msa, scale_msa, gate_msa = shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2)
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2)

        # 注意力模块
        y = modulate(self.norm1(x), shift_msa, scale_msa)
        new_x, attn = self.attention(y, y, y, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(gate_msa * new_x)

        # MLP 模块
        y = modulate(self.norm2(x), shift_mlp, scale_mlp)
        y = self.mlp(y)
        x = x + self.dropout(gate_mlp * y)

        return x, attn

class DiTEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(DiTEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer


    def forward(self, x, c, attn_mask=None, tau=None, delta=None):
        """
        x: 输入张量，形状 (B, L, D)
        c: 条件输入，形状 (B, D)
        attn_mask: 注意力掩码（可选）
        tau, delta: 注意力模块的额外参数（可选）
        返回: (输出张量, 注意力输出列表)
        """
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, c, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class final_layer(nn.Module):
    def __init__(self, d_model, out_channels):
        super(final_layer, self).__init__()
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(d_model, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, 1)
        x = self.linear(modulate(self.norm_final(x),shift,scale))
        return x
