import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Self_layers import AttentionBlock, TimesPatchEmbed
import math
from utils.losses import smooth_loss, mse_loss
from utils.tools import apply_mask

class PatchNTP(nn.Module):
    def __init__(self, configs):
        super(PatchNTP, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len # 必须等于 self.seq_len
        self.kernel = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.c_out = configs.c_out
        self.e_layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout

        self.embedding = TimesPatchEmbed(configs) 

        _attn_ckpt = not getattr(configs, 'no_attn_checkpoint', False)
        # 2. Decoder Stack (保持不变)
        self.decoder = nn.ModuleList(
            [
                AttentionBlock(
                    dim=self.d_model,
                    num_heads=self.n_heads,
                    mlp_ratio=configs.d_ff // self.d_model,
                    dropout=self.dropout,
                    use_checkpoint=_attn_ckpt,
                ) for l in range(self.e_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.d_model)

        # 3. 🎯 关键修改：Patch 解码头 (将 D 维度还原为 K*C_out 维度)
        # 将单个 Patch Token [B, 1, D] 还原成 [B, 1, K * C_out]
        self.patch_decode_head = nn.Linear(self.d_model, self.kernel * self.c_out)


    def forward(self, x_enc: torch.Tensor, mask: torch.Tensor, time_mark: torch.Tensor):
        """
        前向传播方法
        
        Parameters:
            x_enc: 输入编码张量
            mask: 掩码张量
            time_mark: 时间标记张量
            
        Returns:
            训练模式 (self.training=True): dec_out [B, T_seq, C_out]
            评估模式 (self.training=False): (dec_out, all_attns, last_token, last_token_cos)
                - dec_out: 普通输出 [B, T_seq, C_out]
                - last_token: 最后一个token [B, D]
                - all_attns: 所有层的注意力权重列表
                - last_token_cos: 最后一个token与所有token的cos相似度 [B, N]
        """
        x_embed = self.embedding(torch.cat((x_enc, mask.unsqueeze(-1), time_mark), dim=-1)) 
        B, N, D = x_embed.shape
        
        # 3. Encoder Stack (Decoder-only Mode)
        all_attns = []  # 存储所有层的注意力权重
        for layer in self.decoder:
            x_embed, attn = layer(x_embed, is_causal=True, return_attn=not self.training)
            if not self.training:  # 评估模式下收集所有层的attn
                all_attns.append(attn)
     
        x_embed = self.norm(x_embed)
        
        # 评估模式下提取最后一个token
        last_token = None
        last_token_cos = None
        if not self.training:
            last_token = x_embed[:, -1, :]  # [B, D] 最后一个patch token
            # 计算最后一个token与所有token的cos相似度 [B, N]
            try:
                last_token_cos = F.cosine_similarity(x_embed, last_token.unsqueeze(1), dim=-1)
            except Exception as e:
                # 保底处理，避免评估中断
                last_token_cos = torch.zeros(B, N, device=x_embed.device, dtype=x_embed.dtype)
        
        # 4. 🎯 关键修改：Patch 还原 (Temporal Restoration)
        
        # 4a. 将所有 N 个 Patch Token 还原成 K * C_out 的扁平特征
        # dec_out_flat_patches: [B, N, K * C_out]
        dec_out_flat_patches = self.patch_decode_head(x_embed) 
        
        # 4b. 重塑回时序形状 (Sequence Unpatching)
        # 形状: [B, N * K, C_out]
        KC = self.kernel * self.c_out
        dec_out_unpatched = dec_out_flat_patches.reshape(B, N * self.kernel, self.c_out)

        # 4c. 截取到原始序列长度 T (处理 Padding 造成的超长)
        # 目标输出长度 P_pred = T_seq (为了 Shifted Prediction)
        dec_out = dec_out_unpatched[:, :self.seq_len, :]

        # 根据训练模式返回不同的输出
        if not self.training:
            return dec_out, all_attns, last_token, last_token_cos
        else:
            return dec_out

class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.patch_ntp = PatchNTP(configs)
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
            - train/fine-tune: (pred, loss, anomaly_output, imputation_output)
            - pred: (pred, inter_features, inter_attns, imputation_output)
        """
        batch_size, seq_len, bands = x_enc.shape
        if valid_mask is None:
            valid_mask = (1 - torch.isnan(x_enc).int()).to('cuda')

        if mode == 'train':
            # Prepare input

            batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
                x_enc, self.configs.mask_rate, valid_mask
            )
            next_x, next_valid_mask = self._prepare_forecast_label(next_x_enc)
            # Model forward pass
            dec_out = self.patch_ntp(batch_x_masked, mask=missing_mask[:, :, 0].float(), time_mark=time_mark)

            # Calculate loss
            loss = mse_loss(dec_out, next_x, next_valid_mask)

            return dec_out, loss, dec_out, dec_out

        elif mode == 'fine-tune':
             pass

        elif mode == 'pred' or mode == 'test':
            # Prepare input
            batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
                x_enc, self.configs.mask_rate, valid_mask
            )

            # Model forward pass (评估模式下会自动返回额外信息)
            result = self.patch_ntp(
                batch_x_masked, 
                mask=missing_mask[:, :, 0].float(), 
                time_mark=time_mark
            )
            
            # 根据训练模式处理返回值
            if not self.training:
                dec_out, all_attns, last_token, last_token_cos = result
                # Process attention for inter_attns
                inter_features = last_token  # 使用最后一个token作为inter_features
                inter_attns = all_attns     # 使用所有层的attn列表作为inter_attns
                inter_cos = last_token_cos
            else:
                dec_out = result
                all_attns = None
                last_token = None
                inter_cos = None
            
            imputation_output = dec_out

            return dec_out, all_attns, last_token, inter_cos

        else:
            raise ValueError(f"Supported mode: train, fine-tune, or pred")