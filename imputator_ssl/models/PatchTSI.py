from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SaitsEmbedding import SaitsEmbedding
from layers.SAIT_layers import TransformerEncoderLayer, GatedFrequencyFusion
from layers.SelfAttention_Family import ScaledDotProductAttention



class Model(nn.Module):
    def __init__(
        self,
        n_steps: int = 366,
        n_features: int = 3,
        n_layers: int = 2,
        d_model: int = 128,
        n_heads: int = 4,
        d_k: int = 64,
        d_v: int = 64,
        d_ffn: int = 128,
        dropout: float = 0.1,
        attn_dropout: float = 0,
    ):
        super().__init__()

        # concatenate the feature vector and missing mask, hence double the number of features
        actual_n_features = n_features * 2

        self.fft_fusion1 = GatedFrequencyFusion()
        # self.fft_fusion2 = GatedFrequencyFusion()
        # for the 1st block
        self.embedding_1 = SaitsEmbedding(
            n_features,
            d_model,
            with_pos=True,
            n_max_steps=n_steps,
            dropout=dropout,
        )
        self.layer_stack_for_first_block = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    ScaledDotProductAttention(d_k**0.5, attn_dropout),
                    d_model,
                    n_heads,
                    d_k,
                    d_v,
                    d_ffn,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.reduce_dim_z = nn.Linear(d_model, n_features)

        # for the 2nd block
        self.embedding_2 = SaitsEmbedding(
            actual_n_features,
            d_model,
            with_pos=True,
            n_max_steps=n_steps,
            dropout=dropout,
        )

        self.layer_stack_for_second_block = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    ScaledDotProductAttention(d_k**0.5, attn_dropout),
                    d_model,
                    n_heads,
                    d_k,
                    d_v,
                    d_ffn,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.reduce_dim_beta = nn.Linear(d_model, n_features)
        self.reduce_dim_gamma = nn.Linear(n_features, n_features)

        # for delta decay factor
        # self.weight_combine = nn.Linear(n_features + n_steps, n_features)
        self.weight_combine = nn.Linear(n_features + n_steps, n_features)

    def forward(self, x_enc, x_mark, missing_mask, attn_mask: Optional = None) -> Tuple[torch.Tensor, ...]:


        # # first DMSA block
        # enc_out = self.fft_fusion1(x_enc)
        # enc_out = enc_out.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        # mask = missing_mask.permute(0, 2, 1)
        # time_mark = x_mark.permute(0, 2, 1)
        # enc_out = enc_out.unfold(dimension=-1, size=122, step=122) # z: [bs x nvars x patch_num x patch_len]
        # mask = mask.unfold(dimension=-1, size=122, step=122)
        # time_mark = time_mark.unfold(dimension=-1, size=122, step=122)
        # enc_out = enc_out.permute(0, 2, 3, 1).reshape(-1, 122, 3) # intra-period
        # mask = mask.permute(0, 2, 3, 1).reshape(-1, 122, 3)
        # time_mark = time_mark.permute(0, 2, 3, 1).reshape(-1, 122, 2)

        enc_output = self.embedding_1(x_enc)  # namely, term e in the math equation



        first_DMSA_attn_weights = None
        for encoder_layer in self.layer_stack_for_first_block:
            enc_output, first_DMSA_attn_weights = encoder_layer(enc_output, None)

        # enc_output = enc_output.reshape(-1, 366, 128)
        # X_tilde_1 = self.reduce_dim_z(enc_output)
        # X_prime = missing_mask * x_enc + (1 - missing_mask) * X_tilde_1
        #
        #
        # # second DMSA block

        # enc_out = X_prime.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        # enc_out = enc_out.unfold(dimension=-1, size=122, step=122)  # z: [bs x nvars x patch_num x patch_len]
        # enc_out = enc_out.permute(0, 2, 3, 1).reshape(-1, 122, 3)  # intra-period
        # enc_output = self.embedding_2(enc_out,  mask)  # namely term alpha in math algo


        second_DMSA_attn_weights = None
        for encoder_layer in self.layer_stack_for_second_block:
            enc_output, second_DMSA_attn_weights = encoder_layer(enc_output, None)

        # enc_output = enc_output.reshape(-1, 366, 128)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        copy_second_DMSA_weights = second_DMSA_attn_weights.clone()
        copy_second_DMSA_weights = copy_second_DMSA_weights.squeeze(dim=1)  # namely term A_hat in Eq.
        if len(copy_second_DMSA_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            copy_second_DMSA_weights = torch.transpose(copy_second_DMSA_weights, 1, 3)
            copy_second_DMSA_weights = copy_second_DMSA_weights.mean(dim=3)
            copy_second_DMSA_weights = torch.transpose(copy_second_DMSA_weights, 1, 2)
        #
        #
        # # 假设当前 copy_second_DMSA_weights 形状为 (batch_size, seq/3, seq/3)
        # real_batch_size = copy_second_DMSA_weights.shape[0] // 3
        # att_len = copy_second_DMSA_weights.shape[1]  # 当前的注意力大小 (seq/3)
        # seq_len = att_len * 3  # 目标序列长度
        #
        # # 创建完整的注意力矩阵
        # full_attention = torch.zeros(real_batch_size, seq_len, seq_len,
        #                              device=copy_second_DMSA_weights.device)
        #
        # # 填充三个对角块
        # for i in range(3):
        #     start_idx = i * att_len
        #     end_idx = (i + 1) * att_len
        #     batch_start = i * real_batch_size
        #     batch_end = (i + 1) * real_batch_size
        #
        #     # 将对应切片的注意力填充到对角位置
        #     full_attention[:, start_idx:end_idx, start_idx:end_idx] = copy_second_DMSA_weights[batch_start:batch_end]
        #
        # # namely term eta
        # combining_weights = torch.sigmoid(self.weight_combine(torch.cat([missing_mask, full_attention], dim=2)))
        #
        #
        # X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1



        return X_tilde_2, None, copy_second_DMSA_weights