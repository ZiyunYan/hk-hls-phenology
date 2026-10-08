import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
from layers.SaitsEmbedding import SaitsEmbedding



class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.label_len
        self.output_attention = True

        # for the 1st block
        self.embedding = SaitsEmbedding(
            d_in=configs.enc_in+3,
            d_model=configs.d_model,
            with_pos=True,
            embed_type=configs.embed,
            freq=configs.freq,
            n_max_steps=self.pred_len,
            dropout=configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
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

        self.output_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)



    def forward(self, x_enc, x_mark_enc, missing_mask, ano_mask=None):

        enc_out = self.embedding(x_enc, missing_mask[:,:,0].unsqueeze(-1), x_mark_enc)
        enc_out, attns = self.encoder(enc_out)
        dec_out = self.output_projection(enc_out)


        copy_second_DMSA_weights = attns[-1].clone()
        copy_second_DMSA_weights = copy_second_DMSA_weights.squeeze(dim=1)  # namely term A_hat in Eq.
        if len(copy_second_DMSA_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            copy_second_DMSA_weights = torch.transpose(copy_second_DMSA_weights, 1, 3)
            copy_second_DMSA_weights = copy_second_DMSA_weights.mean(dim=3)
            copy_second_DMSA_weights = torch.transpose(copy_second_DMSA_weights, 1, 2)

        return  dec_out, enc_out, enc_out, copy_second_DMSA_weights


