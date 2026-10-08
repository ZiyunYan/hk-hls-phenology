import torch
import torch.nn as nn
import einops
import numpy as np
import math
from utils.losses import mse_loss, smooth_loss, nll_loss, huber_loss, nll_loss_student_t, calculate_nll_t_per_timestep, \
    calculate_nll_per_timestep, dispersive_loss
from utils.tools import apply_mask, process_attention, apply_mask_seasons, sequence2seasons
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from layers.Self_layers import make_2tuple, Swish, RotaryPositionEmbedding1D, PatchEmbedding, ResidualVectorQuantizer, InterSeasonBlock, IntraSeasonBlock, UpsampleBlock, DownsampleBlock
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image


def process_data_tensor(data, mode='sequence', pad_value=0.0):
    """
    处理输入张量 (batch_size, 732, 5) 的函数，支持两种模式。
    
    参数:
    - data: torch.Tensor, 形状 (batch_size, 732, 5)，批量 × 时间步 × 通道。
    - mode: str, 'sequence' 或 'cyclic'。
    - pad_value: float, 填充值，默认为 0.0。
    
    返回:
    - 处理后的 torch.Tensor。
    """
    if not isinstance(data, torch.Tensor):
        raise ValueError("输入数据必须为 torch.Tensor")
    if len(data.shape) != 3 or data.shape[1:] != (732, 5):
        raise ValueError("输入数据形状必须为 (batch_size, 732, 5)")

    batch_size = data.shape[0]

    if mode == 'sequence':
        # 模式1: 纯序列模式 - padding 到 (batch_size, 736, 16)，复制为 3 通道 (batch_size, 736, 16, 3)
        # 时间步 padding: 从 732 到 736，填充 4 行零
        # 通道 padding: 从 5 到 16，填充 11 列零
        padded = torch.nn.functional.pad(
            data, (0, 11, 0, 4), mode='constant', value=pad_value
        )  # [batch_size, 736, 16]
        # 添加通道维度，视为单通道: [batch_size, 736, 16] -> [batch_size, 736, 16, 1]
        padded = padded.unsqueeze(-1)  # [batch_size, 736, 16, 1]
        # 复制 3 次生成 3 通道: [batch_size, 736, 16, 1] -> [batch_size, 736, 16, 3]
        three_channel = padded.repeat(1, 1, 1, 3)  # [batch_size, 736, 16, 3]
        return three_channel

    elif mode == 'cyclic':
        # 模式2: 按周期叠加模式 - 重塑为 (batch_size, 12, 61, 5)，然后 padding 到 (batch_size, 16, 64, 5)
        # 先重塑: 732 = 12 * 61
        reshaped = data.view(batch_size, 12, 61, 5)  # [batch_size, 12, 61, 5]
        # padding: 周期维度 12 -> 16 (填充 4 个零周期)
        # 步骤维度 61 -> 64 (填充 3 个零步骤)
        padded = torch.nn.functional.pad(
            reshaped, (0, 0, 0, 3, 0, 4), mode='constant', value=pad_value
        )  # [batch_size, 16, 64, 5]
        return padded

    else:
        raise ValueError("mode 必须为 'sequence' 或 'cyclic'")

def check_devices(model, input_tensor, name=""):
    """检查模型参数、缓冲区和输入张量的设备"""
    print(f"\n--- Checking devices for {name} ---")
    # 检查输入张量
    print(f"Input tensor device: {input_tensor.device}")
    # 检查模型参数
    for name, param in model.named_parameters():
        if param.device != torch.device('cuda:0'):
            print(f"Parameter {name} on {param.device}")
    # 检查模型缓冲区（如 LayerNorm 的 weight 和 bias）
    for name, buffer in model.named_buffers():
        if buffer.device != torch.device('cuda:0'):
            print(f"Buffer {name} on {buffer.device}")

class Model(nn.Module):
    """
    Vanilla Transformer with DINOv3 backbone for image processing.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.dropout = nn.Dropout(p=0.2)
        print('mask ratio:', configs.mask_rate)

        # Load DINOv3 model and processor
        pretrained_model_name = "facebook/dinov3-vitl16-pretrain-sat493m"
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.imputation_model = AutoModel.from_pretrained(
            pretrained_model_name,
            device_map="cuda:0",
        )
        # print('imputation_model.device', self.imputation_model.device)
    def _prepare_forecast_label(self, next_x):
        """Prepare input tensor with padding and masking for forecasting."""
        next_valid_mask = (1 - torch.isnan(next_x).int()).to(next_x.device)

        # Apply mask
        next_batch, _, _, _ = apply_mask(
            next_x, next_valid_mask, 0, next_x.device
        )

        return next_batch, next_valid_mask

    def _prepare_input(self, x_enc, mask_ratio, valid_mask=None):
        """Prepare input image tensor with padding and masking."""
        if valid_mask is None:
            valid_mask = (1 - torch.isnan(x_enc).int()).to(x_enc.device)
        print('x_enc.device', x_enc.device)
        print('valid_mask.device', valid_mask.device)
        print('mask_ratio', mask_ratio)
        # Apply mask
        batch_x, batch_x_masked, missing_mask, indicating_mask = apply_mask(
            x_enc, valid_mask, mask_ratio, x_enc.device
        )

        return batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask

    def forward(self, x_enc, time_mark=None, valid_mask=None, next_x_enc=None, mode='train'):
        """
        General forward pass for the model, supporting train, fine-tune, and predict modes.

        Parameters:
            x_enc: Input image tensor, shape (batch_size, channels, height, width)
            time_mark: Time mark, defaults to None (not used for images)
            valid_mask: Valid mask, defaults to None
            next_x_enc: Target data for forecast mode, defaults to None
            mode: Mode of operation, 'train', 'fine-tune', or 'pred'

        Returns:
            Depending on mode:
            - train/fine-tune: (pred, loss, anomaly_output, imputation_output)
            - pred: (pred, inter_features, inter_attns, imputation_output)
        """

        if valid_mask is None:
            valid_mask = (1 - torch.isnan(x_enc).int()).to(x_enc.device)

        if mode == 'train':
            # Prepare input
            batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
                x_enc, self.configs.mask_rate, valid_mask
            )

            # Process masked input through DINOv3
            inputs = self.processor(images=batch_x_masked, return_tensors="pt", do_rescale=False)
            inputs = {k: v.to(self.imputation_model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.imputation_model(**inputs)

            # Extract outputs (assuming pooler_output as the main feature)
            dec_out = outputs.pooler_output  # Shape: (batch_size, hidden_size)
            enc_out = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
            atten = outputs.attentions[-1] if outputs.attentions is not None else None  # Last layer attention

            # Calculate loss
            # Reshape batch_x to match dec_out for loss calculation (adjust as needed)
            target = batch_x.view(batch_size, -1)  # Flatten for simplicity
            dec_out_reshaped = dec_out.view(batch_size, -1)  # Adjust shape if necessary
            loss = cal_rec_loss(dec_out_reshaped, target, indicating_mask, alpha=1, beta=0.5)

            return dec_out, loss, enc_out, dec_out

        elif mode == 'fine-tune':
            # Similar to train mode, adjust as needed
            batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
                x_enc, self.configs.mask_rate, valid_mask
            )
            inputs = self.processor(images=batch_x_masked, return_tensors="pt", do_rescale=False)
            inputs = {k: v.to(self.imputation_model.device) for k, v in inputs.items()}

            outputs = self.imputation_model(**inputs)
            dec_out = outputs.pooler_output
            enc_out = outputs.last_hidden_state
            atten = outputs.attentions[-1] if outputs.attentions is not None else None

            target = batch_x.view(batch_size, -1)
            dec_out_reshaped = dec_out.view(batch_size, -1)
            loss = cal_rec_loss(dec_out_reshaped, target, indicating_mask, alpha=1, beta=0.5)

            return dec_out, loss, enc_out, dec_out

        elif mode == 'pred':
            # Prediction mode: no loss calculation
            batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
                x_enc, 0.0, valid_mask  # No masking for prediction
            )

            batch_x_masked = process_data_tensor(batch_x_masked, mode='sequence')
            inputs = {"pixel_values": batch_x_masked.permute(0, 3, 1, 2)}
            check_devices(self.imputation_model, batch_x_masked, "imputation_model forward pass")
            with torch.inference_mode():
                outputs = self.imputation_model(**inputs)

            dec_out = outputs.pooler_output
            enc_out = outputs.last_hidden_state
            # atten = outputs.attentions[-1] if outputs.attentions is not None else None
            # print('dec_out.shape', dec_out.shape)
            # print('enc_out.shape', enc_out.shape)
            # print('atten.shape', atten.shape)

            return enc_out[:,0,:], enc_out[:,5:,:], dec_out, dec_out
