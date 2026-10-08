__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp
from utils.losses import smooth_loss, mse_loss
from utils.tools import apply_mask

def cal_rec_loss(pred, target, mask, alpha, beta):

    rec_loss = mse_loss(pred, target, mask)

    smo_loss = smooth_loss(pred, mode='dy2')

    return alpha * rec_loss + beta * smo_loss

class Model(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 244, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):

        super().__init__()
        self.configs = configs
        self.dropout = nn.Dropout(p=0.2)
        print('mask ratio:', configs.mask_rate)
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.label_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        decomposition = configs.decomposition
        kernel_size = configs.kernel_size



        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                                 patch_len=patch_len, stride=stride,
                                                 max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                 n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                 attn_dropout=attn_dropout,
                                                 dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                 padding_var=padding_var,
                                                 attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                                 store_attn=store_attn,
                                                 pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                 head_dropout=head_dropout, padding_patch=padding_patch,
                                                 pretrain_head=pretrain_head, head_type=head_type,
                                                 individual=individual, revin=revin, affine=affine,
                                                 subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                               patch_len=patch_len, stride=stride,
                                               max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                               n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                               attn_dropout=attn_dropout,
                                               dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                               padding_var=padding_var,
                                               attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                               store_attn=store_attn,
                                               pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                               head_dropout=head_dropout, padding_patch=padding_patch,
                                               pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                               revin=revin, affine=affine,
                                               subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                           patch_len=patch_len, stride=stride,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                           n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                           attn_dropout=attn_dropout,
                                           dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                           padding_var=padding_var,
                                           attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                           store_attn=store_attn,
                                           pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                                           padding_patch=padding_patch,
                                           pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                           revin=revin, affine=affine,
                                           subtract_last=subtract_last, verbose=verbose, **kwargs)

        
    
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
            time_mark = self.dropout(time_mark) if time_mark is not None else None

            # Model forward pass
            if self.decomposition:
                res_init, trend_init = self.decomp_module(batch_x_masked)
                res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)  # [Batch, Channel, Input length]
                res = self.model_res(res_init)
                trend = self.model_trend(trend_init)
                dec_out = res + trend
                dec_out = dec_out.permute(0, 2, 1)  # [Batch, Input length, Channel]
                dec_out = dec_out[:,:,:self.configs.c_out]
            else:
                
                enc_in = torch.cat((batch_x_masked, time_mark, missing_mask[:,:,0].unsqueeze(-1)), dim=-1)
                enc_in = enc_in.permute(0, 2, 1)  # [Batch, Channel, Input length]
                output = self.model(enc_in)
                dec_out = output.permute(0, 2, 1)  # [Batch, Input length, Channel]
                dec_out = dec_out[:,:,:self.configs.c_out]
            # Calculate loss
            loss = cal_rec_loss(dec_out, batch_x, indicating_mask, 1, 0.5)
            return dec_out, loss, dec_out, dec_out

        elif mode == 'fine-tune':
            # Randomize mask_rate
            mask_rate = 0 if np.random.rand() < 0.5 else np.random.uniform(0, 0.8)
            # Prepare input
            batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
                x_enc, self.configs.mask_rate, valid_mask
            )
            time_mark = self.dropout(time_mark) if time_mark is not None else None

            # Model forward pass
            if self.decomposition:
                res_init, trend_init = self.decomp_module(batch_x_masked)
                res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
                res = self.model_res(res_init)
                trend = self.model_trend(trend_init)
                dec_out = res + trend
                dec_out = dec_out.permute(0, 2, 1)
                dec_out = dec_out[:,:,:self.configs.c_out]
            else:
                enc_in = torch.cat((batch_x_masked, time_mark, missing_mask[:,:,0].unsqueeze(-1)), dim=-1)
                enc_in = enc_in.permute(0, 2, 1)
                output = self.model(enc_in)
                dec_out = output.permute(0, 2, 1)
                dec_out = dec_out[:,:,:self.configs.c_out]
            # Calculate loss based on mask_rate
            loss = 0
            if mask_rate == 0:
                if next_x_enc is not None:
                    next_x, next_valid_mask = self._prepare_forecast_label(next_x_enc)
                    loss += cal_rec_loss(dec_out, next_x[:, :seq_len, :], next_valid_mask, 1, 0.5)
                loss += cal_rec_loss(dec_out, batch_x, missing_mask, 1, 0.5)
            else:
                loss += cal_rec_loss(dec_out, batch_x, indicating_mask, 1, 0.5)

            return dec_out, loss, dec_out, dec_out

        elif mode == 'pred' or mode == 'test':
            # Prepare input
            batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
                x_enc, self.configs.mask_rate, valid_mask
            )
            time_mark = self.dropout(time_mark) if time_mark is not None else Nonee

            # Model forward pass
            if self.decomposition:
                res_init, trend_init = self.decomp_module(batch_x_masked)
                res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
                res = self.model_res(res_init)
                trend = self.model_trend(trend_init)
                dec_out = res + trend
                dec_out = dec_out.permute(0, 2, 1)
                dec_out = dec_out[:,:,:self.configs.c_out]
            else:
                enc_in = torch.cat((batch_x_masked, time_mark, missing_mask[:,:,0].unsqueeze(-1)), dim=-1)
                enc_in = enc_in.permute(0, 2, 1)
                output = self.model(enc_in)
                dec_out = output.permute(0, 2, 1)
                dec_out = dec_out[:,:,:self.configs.c_out]
            # Process attention for inter_attns
            inter_features = None  # Placeholder, as original model doesn't return this
            inter_attns = None    # Placeholder, as original model doesn't return this
            imputation_output = dec_out

            # return dec_out, inter_features, inter_attns, imputation_output
            return dec_out, dec_out, dec_out, imputation_output

        else:
            raise ValueError(f"Supported mode: train, fine-tune, or pred")