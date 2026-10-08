import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
from layers.Embedding import Embedding
from layers.Self_layers import AttentionBlock
from utils.losses import smooth_loss, mse_loss
from utils.tools import apply_mask


class Imputator(nn.Module):
    """
    Vanilla Transformer Encoder using RoPE and CLS/Register Tokens.
    """
    def __init__(self, configs):
        super(Imputator, self).__init__()
        self.pred_len = configs.label_len
        self.seq_len = configs.seq_len
        self.output_attention = True
        self.dropout_rate = configs.dropout # 存储 Dropout 率
        
        # CLS Tokens and Position Embeddings (保持不变)
        self.cls_token = nn.Parameter(torch.randn(1, 1, configs.d_model))
        self.cls_pos_embed = nn.Parameter(torch.randn(1, 1, configs.d_model))
        
        # 🎯 关键修改 1: 移除绝对位置编码 (APE)
        self.embedding = Embedding(
            d_in=configs.enc_in + 4,
            d_model=configs.d_model,
            with_pos=True, 
            embed_type=configs.embed,
            n_max_steps=self.seq_len,
        )

        
        _attn_ckpt = not getattr(configs, 'no_attn_checkpoint', False)
        self.encoder_layers = nn.ModuleList(
            [
                AttentionBlock(
                    dim=configs.d_model,
                    num_heads=configs.n_heads,
                    mlp_ratio=configs.d_ff // configs.d_model, # 使用 d_ff/d_model 计算 mlp_ratio
                    dropout=configs.dropout,
                    use_checkpoint=_attn_ckpt,
                ) for l in range(configs.e_layers)
            ]
        )
        self.norm = torch.nn.LayerNorm(configs.d_model) # 最终 Layer Norm

        self.output_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)


    def forward(self, x_enc, mask=None, time_mark=None):
        """
        前向传播。
        
        Returns:
            dec_out: 模型输出 [B, T, C_out]
            processed_attns_list: 处理后的注意力权重列表（训练时为None，测试时为列表，包含所有层的注意力）
            cls_out: CLS token的输出 [B, D]
            cls_similarity: CLS token与序列的余弦相似度 [B, T]（训练时为None，测试时为张量）
        """
        # 1. 嵌入输入数据 (不含 APE)
        enc_out = self.embedding(x_enc, mask.unsqueeze(-1), time_mark)

        # 2. 拼接 CLS 令牌
        cls_tokens = self.cls_token.expand(enc_out.shape[0], -1, -1)
        cls_tokens = cls_tokens + self.cls_pos_embed
        enc_out = torch.cat((cls_tokens, enc_out), dim=1) # Shape: (B, T+1, D)

        # 3. 编码器前向传播 (使用 ModuleList)
        attns_list = []
        for layer in self.encoder_layers:
            enc_out, attn = layer(enc_out, return_attn=not self.training)
            if not self.training:
                attns_list.append(attn)
        
        enc_out = self.norm(enc_out) # 最终归一化

        # 4. 提取特征和投影 (保留了 CLS/Seq 分离和 Dropout)
        cls_out = enc_out[:, 0, :]
        seq_out = enc_out[:, 1:, :]
        
        seq_out = F.dropout(seq_out, p=self.dropout_rate, training=self.training)
        dec_out = self.output_projection(seq_out)

        if self.training:
            # 训练阶段：不计算 cls_similarity 与 processed_attns，节省显存和计算
            processed_attns_list = None
            cls_similarity = None
        else:
            # 仅在评估/特征提取阶段计算相似度与处理后的 Attention
            cls_vec = cls_out.unsqueeze(1)  # [B, 1, D]
            seq_vec = seq_out  # [B, T, D]

            # L2 归一化 (计算余弦相似度)
            cls_vec_norm = F.normalize(cls_vec, p=2, dim=-1)
            seq_vec_norm = F.normalize(seq_vec, p=2, dim=-1)

            # 计算相似度：(B, 1, D) @ (B, D, T) -> (B, 1, T)
            # 注意：这里使用 seq_vec_norm.transpose(-1, -2) 相当于计算 Q @ K^T
            cls_similarity = torch.bmm(cls_vec_norm, seq_vec_norm.transpose(-1, -2)).squeeze(1)
            # 最终 shape: (B, T)

            # ============================================================
            # 处理 Attention List：去掉 CLS 令牌对应的行列
            # ============================================================
            processed_attns_list = []
            for attn in attns_list:
                # 去掉CLS token对应的行列，只保留序列token之间的注意力
                attn_no_cls = attn[:, 1:, 1:]   # Shape: [Batch, T, T]
                processed_attns_list.append(attn_no_cls)

        # 统一返回格式：训练时 processed_attns_list / cls_similarity 为 None
        # 返回：dec_out, processed_attns_list, cls_out, cls_similarity
        return dec_out, processed_attns_list, cls_out, cls_similarity

class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.imputation_model = Imputator(configs)
        self.configs = configs
        print('mask ratio:', configs.mask_rate)

    def _prepare_forecast_label(self, next_x):
        """Prepare input tensor with padding and masking."""

        next_valid_mask = (1 - torch.isnan(next_x).int()).to('cuda')

        # Apply mask
        next_batch, _, _, _ = apply_mask(
            next_x, next_valid_mask, 0, 'cuda'
        )

        return next_batch, next_valid_mask

    def _prepare_input(self, x_enc, mask_ratio, valid_mask=None):
        """Prepare input tensor with padding and masking."""
        # batch_size, seq_len, bands = x_enc.shape
        if valid_mask is None:
            valid_mask = (1 - torch.isnan(x_enc).int()).to('cuda')

        # Apply mask
        batch_x, batch_x_masked, missing_mask, indicating_mask = apply_mask(
            x_enc, valid_mask, mask_ratio, 'cuda'
        )

        return batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask

    def forward(self, x_enc, time_mark=None, valid_mask=None, next_x_enc=None, mode='train'):
        """
        General forward pass for the model, supporting train, fine-tune, and predict modes.

        Parameters:
            x_enc: Input encoding, shape (batch_size, seq_len, feature_dim)
            time_mark: Time mark, defaults to None
            valid_mask: Valid mask, defaults to None
            next_x_enc: Target data for forecast mode, defaults to None
            mode: Mode of operation, 'train', 'fine-tune', or 'pred'

        Returns:
            Depending on mode:
            - train: (dec_out, loss, attens, cls_out)
                - dec_out: 模型输出 [B, T, C_out]
                - loss: 损失值
                - attens: None（训练时不保存注意力）
                - cls_out: CLS token输出 [B, D]
            - pred/test: (dec_out, attens, cls_out, cls_similarity)
                - dec_out: 模型输出 [B, T, C_out]
                - attens: 所有层的注意力权重列表（已去除CLS token）
                - cls_out: CLS token输出 [B, D]
                - cls_similarity: CLS token与序列的余弦相似度 [B, T]
        """
        batch_size, seq_len, bands = x_enc.shape
        if valid_mask is None:
            valid_mask = (1 - torch.isnan(x_enc).int()).to('cuda')

        if mode == 'train':
            # Prepare input

            batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
                x_enc, self.configs.mask_rate, valid_mask
            )

            # Model forward pass
            dec_out, attens, cls_out, cls_similarity = self.imputation_model(batch_x_masked, mask=missing_mask[:, :, 0].float(), time_mark=time_mark)

            # Calculate loss
            loss = mse_loss(dec_out, batch_x, indicating_mask)

            return dec_out, loss, attens, cls_out

        elif mode == 'fine-tune':
             pass

        elif mode == 'pred' or mode == 'test':
            # Prepare input
            batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
                x_enc, self.configs.mask_rate, valid_mask
            )

            # Model forward pass
            # 返回：dec_out, attns (所有层的注意力权重列表), cls_out, cls_similarity (CLS与序列的余弦相似度)
            dec_out, attens, cls_out, cls_similarity = self.imputation_model(batch_x_masked, mask=missing_mask[:, :, 0].float(), time_mark=time_mark)

            return dec_out, attens, cls_out, cls_similarity

        else:
            raise ValueError(f"Supported mode: train, fine-tune, or pred")




