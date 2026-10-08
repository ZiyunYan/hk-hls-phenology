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


class Transformer_NTP(nn.Module):
    """
    Vanilla Transformer decoder (Sequence to Sequence) for Next Point Prediction (NPP).
    """
    def __init__(self, configs):
        super(Transformer_NTP, self).__init__()
        self.pred_len = configs.label_len
        self.seq_len = configs.seq_len
        self.output_attention = True
        self.dropout_rate = configs.dropout

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
                    mlp_ratio=configs.d_ff // configs.d_model,
                    dropout=configs.dropout,
                    use_checkpoint=_attn_ckpt,
                ) for l in range(configs.e_layers)
            ]
        )
        self.norm = torch.nn.LayerNorm(configs.d_model)

        # 2. 输出投影层：将 [B, T, D] 映射到 [B, T, C_out]
        self.output_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)


    def forward(self, x_enc: torch.Tensor, mask: torch.Tensor, time_mark: torch.Tensor):
        

        enc_out = self.embedding(x_enc, mask.unsqueeze(-1), time_mark)
        
        attns_list = []
        for layer in self.encoder_layers:
            enc_out, attn = layer(enc_out, is_causal=True, return_attn=not self.training)
            attns_list.append(attn)
        
        enc_out = self.norm(enc_out) 


        seq_out = F.dropout(enc_out, p=self.dropout_rate, training=self.training)
        
        dec_out = self.output_projection(seq_out)

        
        dec_out = dec_out[:, :self.pred_len, :] # 截取到预测长度 (通常 pred_len == seq_len)
        
        return dec_out, attns_list

class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.transformer_ntp = Transformer_NTP(configs)
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
            dec_out, _ = self.transformer_ntp(batch_x_masked, mask=missing_mask[:, :, 0].float(), time_mark=time_mark)

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

            # Model forward pass
            dec_out, atten, enc_out = self.transformer_ntp(batch_x_masked, mask=missing_mask[:, :, 0].float(), time_mark=time_mark)

            # Process attention for inter_attns
            inter_features = enc_out  # Use enc_out as inter_features
            inter_attns = atten       # Use atten as inter_attns
            imputation_output = dec_out

            return dec_out, inter_features, inter_attns, imputation_output

        else:
            raise ValueError(f"Supported mode: train, fine-tune, or pred")




