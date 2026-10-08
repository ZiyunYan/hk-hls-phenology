import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgen.native_function_generation import self_to_out_signature

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np
from layers.SaitsEmbedding import SaitsEmbedding


def anomaly_prob_to_mask(anomaly_prob, mode='predict', threshold=0.8):
    """
    将异常概率转换为二元掩码。

    参数:
    anomaly_prob (torch.Tensor): 形状为 (batch, steps, 1) 的张量，表示每个时间步的异常概率。
    mode (str): 'sampling' 或 'predict'，决定使用哪种模式来生成掩码。
    threshold (float): 在 'predict' 模式下使用的阈值，默认为 0.8。

    返回:
    torch.Tensor: 形状为 (batch, steps) 的二元掩码，其中 1 表示异常（掩膜），0 表示正常（不掩膜）。
    """

    if mode not in ['sampling', 'predict']:
        raise ValueError("Mode must be either 'sampling' or 'predict'")

    anomaly_prob = anomaly_prob[:, :, 0]  # 去掉最后一个维度，变为 (batch, steps)

    if mode == 'sampling':
        # 直接使用概率进行采样
        mask = torch.bernoulli(anomaly_prob)
    else:  # mode == 'predict'
        # 使用阈值进行预测，概率高于阈值被认为是异常
        mask = (anomaly_prob > threshold).float()

    return mask

class AnomalyDetector(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers):
        super(AnomalyDetector, self).__init__()

        self.embedding =  SaitsEmbedding(
            d_in=4,
            d_model=64,
            with_pos=True,
            n_max_steps=366,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 双向LSTM的输出维度是hidden_size * 2
        lstm_output_size = hidden_size * 2

        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)

        # 异常概率输出头
        self.anomaly_head = nn.Sequential(nn.Linear(64, 1),
                                          nn.Sigmoid())


    def forward(self, x):
        # x shape: (batch, steps, input_size)
        # 如果使用嵌入层，取消下面这行的注释
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.fc2(x)
        # 异常概率输出
        anomaly_prob = self.anomaly_head(x)
        # shape: (batch, steps, 1)
        # Value输出

        return anomaly_prob

def normalized_masked_error(coarse_fill, x_enc, missing_mask):
    """
    计算归一化的掩码误差，每个通道单独归一化，并考虑缺失值。

    参数:
    coarse_fill: 预测值张量，形状为 (batch_size, length, bands)
    x_enc: 真实值张量，形状为 (batch_size, length, bands)
    missing_mask: 掩码张量，形状为 (batch_size, length, bands)，值为 1 表示有效，0 表示缺失

    返回:
    归一化后的总误差张量，形状为 (batch_size, length, 1)
    """

    # 计算绝对误差
    abs_errors = torch.abs(coarse_fill - x_enc)

    # 应用掩码：只保留有效位置的误差
    masked_errors = abs_errors * missing_mask

    # 计算每个通道的均值和标准差，考虑掩码
    # 我们使用masked_errors进行计算，确保只在有效值上计算均值和标准差
    masked_errors_sum = torch.sum(masked_errors, dim=1, keepdim=True)
    mask_count = torch.sum(missing_mask, dim=1, keepdim=True)

    # 计算均值
    mean = masked_errors_sum / mask_count

    # 计算标准差
    variance = torch.sum((masked_errors - mean) ** 2 * missing_mask, dim=1, keepdim=True) / mask_count
    std = torch.sqrt(variance)

    # 避免除以零
    std = torch.clamp(std, min=1e-8)

    # 使用 z-score 标准化
    normalized_errors = (masked_errors - mean) / std

    # 确保掩码外的值为 0
    normalized_errors = normalized_errors * missing_mask

    # 沿通道维度求和并增加一个维度
    total_error = torch.sum(normalized_errors, dim=-1, keepdim=True)

    return total_error




class Transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Transformer, self).__init__()
        self.pred_len = configs.label_len
        self.output_attention = True
        self.beta = 1.0  # sigmoid的平滑度参数

        # for the 1st block
        self.SaitsEmded1 = SaitsEmbedding(
            d_in=configs.enc_in,
            d_model=configs.d_model,
            with_pos=True,
            embed_type=configs.embed,
            freq=configs.freq,
            n_max_steps=self.pred_len,
            dropout=0.1,
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



    def forward(self, x_enc, x_mark_enc, missing_mask):


        enc_out = self.SaitsEmded1(x_enc)
        enc_out, attns = self.encoder(enc_out)

        dec_out = self.output_projection(enc_out)


        copy_second_DMSA_weights = attns[-1].clone()
        copy_second_DMSA_weights = copy_second_DMSA_weights.squeeze(dim=1)  # namely term A_hat in Eq.
        if len(copy_second_DMSA_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            copy_second_DMSA_weights = torch.transpose(copy_second_DMSA_weights, 1, 3)
            copy_second_DMSA_weights = copy_second_DMSA_weights.mean(dim=3)
            copy_second_DMSA_weights = torch.transpose(copy_second_DMSA_weights, 1, 2)

        return  dec_out, copy_second_DMSA_weights



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.imputation_model = Transformer(configs)
        self.anomaly_model = AnomalyDetector()

    def forward(self, x_enc, x_mark_enc, missing_mask):
        reconstructed_x, atten = self.imputation_model(x_enc, x_mark_enc, missing_mask)
        anaomaly_prob = self.anomaly_model(reconstructed_x)

        return reconstructed_x, atten, anaomaly_prob, None