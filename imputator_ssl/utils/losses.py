# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

from __future__ import annotations

import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb
import warnings



def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class TruncatedExponentialKL:
    def __init__(self, lambda_param=15):
        self.lambda_param = t.tensor(lambda_param)  # λ控制稀疏强度
        self.Z = 1 - t.exp(-self.lambda_param)  # 归一化常数修正

    def log_prob(self, x):
        """
        x: anomaly_prob ∈ [0,1]（需通过Sigmoid约束）
        """
        # 计算截断指数分布的对数概率
        log_Z = t.log(self.Z)
        log_p = t.log(self.lambda_param) - self.lambda_param * x - log_Z

        # 强制约束x ∈ [0,1]，否则概率为0（log_p=-inf）
        invalid_mask = (x < 0) | (x > 1)
        log_p = t.where(invalid_mask, -t.inf, log_p)
        return log_p


def compute_spike_score(pred, window_size=7, current_weight=0.1, decay_rate=0.9):
    batch_size, seq_len, bands = pred.shape
    assert window_size % 2 == 1, "window_size 必须为奇数"
    pad = window_size // 2

    # 边缘填充（沿时间维度）
    padded_pred = t.nn.functional.pad(pred, (0, 0, pad, pad), mode='replicate')

    # 生成滑动窗口
    windows = padded_pred.unfold(dimension=1, size=window_size, step=1)

    # 修正后的权重计算（含设备一致性）
    distances = t.arange(window_size, device=pred.device).float() - pad
    distance_weights = t.exp(-decay_rate * t.abs(distances))
    distance_weights = distance_weights / distance_weights.sum()

    center_idx = pad
    neighbor_total_weight = 1 - current_weight
    denominator = (distance_weights.sum() - distance_weights[center_idx]) + 1e-8  # 防止除零
    neighbor_weights = distance_weights * neighbor_total_weight / denominator
    adjusted_weights = neighbor_weights.clone()
    adjusted_weights[center_idx] = current_weight

    # 权重形状调整
    adjusted_weights = adjusted_weights.view(1, 1, 1, window_size).expand(-1, -1, bands, -1)

    # 计算动量与尖峰得分
    momentum = t.sum(windows * adjusted_weights, dim=-1)
    spike_score = t.abs(pred - momentum).mean(dim=-1)

    return spike_score

class CustomLoss(nn.Module):
    def __init__(self, missing_weight=1, indicating_weight=1, smooth_weight=1, indicating_weight2=0.1, bce_weight = 1):
        super().__init__()
        self.missing_weight = missing_weight
        self.indicating_weight = indicating_weight
        self.smooth_weight = smooth_weight
        self.indicating_weight2 = indicating_weight2
        self.anomaly_bce_weight = bce_weight
        self.eps = 1e-8


    def _compute_masked_mse(self, pred, target, mask, prob=None):
        # # 只检查mask为1的位置的target值是否有0
        # masked_target = target[mask.bool()]
        # assert not t.any(masked_target == 0), "Found zeros in masked target values"
        if prob is not None:
            loss = (pred - target)**2 * mask * (1-prob)
        else:
            loss = (pred - target)**2 * mask
        if prob is not None:
            mask_sum = t.sum((1-prob)*mask)
        else:
            mask_sum = mask.sum()
        return t.where(mask_sum > 0,
                       loss.sum() / mask_sum,
                       t.tensor(0.0, device=pred.device))

    def _compute_smoothness_loss(self, pred, mode=None):
        """计算改进的二阶差分损失，强调方向变化"""
        # 首先检查输入是否包含 NaN/Inf
        if t.isnan(pred).any() or t.isinf(pred).any():
            warnings.warn("Found NaN/Inf in prediction tensor")
            return t.tensor(0.0, device=pred.device)
        # 计算一阶差分
        dy = pred[:, 1:] - pred[:, :-1]  # dy[i] = pred[i+1] - pred[i], 长度N-1
        # 计算二阶差分
        d2y = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]  # 长度N-2
        if mode == 1:
            loss = t.mean(dy ** 2)
        else:
            loss = t.mean(d2y ** 2)

        return loss

    def _binary_cross_entropy_with_temperature(
            self,
            anomaly_prob,  # 模型预测的异常概率 [batch, seq_len, 1]
            anomaly_mask,  # 真实异常标签 [batch, seq_len, 1]
            initial_pred,
            final_pred,  # 模型重建值 [batch, seq_len]
            target,  # 真实值 [batch, seq_len]
            missing_mask,
            temperature=1,
            alpha=1,  # 真实标签权重
            recon_weight=0.5,
            spike_weight=0.5,
            window_size=5,  # 动量窗口大小
            current_weight=0.1
    ):
        def minmax_norm(x, mask):
            """
            仅在有效值（mask=1）上计算最小值和最大值，并进行归一化
            Args:
                x: [batch, steps] 输入数据
                mask: [batch, steps] 掩码（0表示无效，1表示有效）
            Returns:
                normalized_x: [batch, steps] 归一化后的数据
            """
            # 将无效值替换为 inf，以便在计算最小值和最大值时忽略
            x_masked = x.masked_fill(mask == 0, float('inf'))

            # 计算有效值的最小值
            min_val = x_masked.min(dim=1, keepdim=True)[0]

            # 将无效值替换为 -inf，以便在计算最大值时忽略
            x_masked = x.masked_fill(mask == 0, float('-inf'))

            # 计算有效值的最大值
            max_val = x_masked.max(dim=1, keepdim=True)[0]

            # 归一化
            normalized_x = (x - min_val) / (max_val - min_val + 1e-6)

            # 将无效值的位置重置为 0（或其他默认值）
            normalized_x = normalized_x.masked_fill(mask == 0, 0)

            return normalized_x

        missing_mask = missing_mask[:,:,0]
        # 1. 重建误差概率
        reconstruction_error = t.mean((final_pred - target) ** 2, dim=-1)  # [batch, seq_len]
        # print('reconstruction_error', t.isnan(reconstruction_error).any())
        recon_prob = minmax_norm(reconstruction_error, missing_mask)  # [batch, seq_len]

        # 2. 尖峰得分概率
        spike_score = compute_spike_score(final_pred, window_size, current_weight)  # [batch, seq_len]
        spike_prob = minmax_norm(spike_score, missing_mask)  # [batch, seq_len]

        # 3. 融合真实标签和补充概率
        combined_mask = (
                anomaly_mask.squeeze(-1) * alpha +
                recon_prob * recon_weight +
                spike_prob * spike_weight
        ).unsqueeze(-1)  # [batch, seq_len, 1]
        combined_mask = t.clamp(combined_mask, 0, 1)
        # print('combined_mask', t.isnan(combined_mask).any())
        # print('recon_prob', t.isnan(recon_prob).any())
        # print('anomaly_mask', t.isnan(anomaly_mask).any())
        # print('spike_prob', t.isnan(spike_prob).any())
        # 应用温度缩放
        scaled_probs = anomaly_prob ** (1 / temperature)
        # print('scaled_probs', t.isnan(scaled_probs).any())
        # 计算二元交叉熵损失
        loss = F.binary_cross_entropy(scaled_probs, combined_mask, reduction='none')

        avg_loss = t.sum(loss * missing_mask.unsqueeze(-1)) / (t.sum(missing_mask.unsqueeze(-1)) + 1e-6)

        return avg_loss



    def forward(self, initial_pred, target, final_pred, anomaly_prob, missing_mask=None, indicating_mask=None, anomaly_mask=None, smooth=False, missing=False, indicating=False):
        # anomaly_prob = None
        final_pred =None
        # summ = t.sum(anomaly_mask)
        # for vqvae --- dec_out, vq_loss, cls_out,


        # 计算两部分的 MAE
        if missing:
            missing_loss = self._compute_masked_mse(initial_pred, target, missing_mask)
            # missing_loss = self._compute_masked_mse_sam(initial_pred, target, missing_mask)
        else:
            missing_loss = 0
        if indicating:
            indicating_loss = self._compute_masked_mse(initial_pred, target, indicating_mask)
            # indicating_loss = self._compute_masked_mse_sam(initial_pred, target, indicating_mask)
        else:
            indicating_loss = 0

        if smooth:
            smoothness_loss1 = self._compute_smoothness_loss(initial_pred)
            # smoothness_loss2 = self._compute_smoothness_loss(final_pred)
            smoothness_loss = smoothness_loss1
        else:
            smoothness_loss = 0
        if final_pred is not None:
            # final_indicating_loss = self._compute_masked_mse(final_pred, target, indicating_mask)
            final_indicating_loss = final_pred
        else:
            final_indicating_loss = 0
        if anomaly_prob is not None:
            # anomalies_bce_loss = self._binary_cross_entropy_with_temperature(anomaly_prob,anomaly_mask.unsqueeze(-1), initial_pred.detach(), initial_pred.detach(), target, missing_mask)
            anomalies_bce_loss = anomaly_prob
        else:
            anomalies_bce_loss = 0
        # reconstruction_correlation_loss = self._reconstruction_correlation_loss(pred, target, anomaly_prob, missing_mask)
        # 组合所有loss
        total_loss = (self.missing_weight * missing_loss +
                      self.indicating_weight * indicating_loss +
                     self.smooth_weight * smoothness_loss +
                      self.indicating_weight2 * final_indicating_loss +
                      self.anomaly_bce_weight * anomalies_bce_loss)


        return total_loss


def patch_mse_loss(pred_patches, target_patches, patch_mask=None):
    """
    对 patch 级别的重建计算 MSE loss。
    用于掩码自监督等场景：pred_patches/target_patches 形状 [B, N, patch_len*c_out]，
    patch_mask 形状 [B, N]，1 表示该 patch 参与 loss 计算。

    Args:
        pred_patches: 预测的 patch [B, N, P]
        target_patches: 目标 patch [B, N, P]
        patch_mask: 可选，[B, N]，1 表示参与计算

    Returns:
        torch.Tensor: 标量 MSE loss
    """
    if patch_mask is None:
        return ((pred_patches - target_patches) ** 2).mean()
    patch_mask = patch_mask.unsqueeze(-1).float()
    diff = (pred_patches - target_patches) ** 2
    masked_diff = diff * patch_mask
    num_valid = patch_mask.sum().clamp(min=1)
    return masked_diff.sum() / num_valid


def mse_loss(pred, target, mask=None):
    """
    Compute masked Mean Squared Error (MSE) loss.

    Args:
        pred (torch.Tensor): Predicted values, shape [batch_size, seq_len, target_channels] or similar.
        target (torch.Tensor): Target values, same shape as pred.
        mask (torch.Tensor, optional): Mask tensor, shape [batch_size, seq_len] or broadcastable.
                                      1 for positions to include, 0 to exclude. If None, no masking.

    Returns:
        torch.Tensor: Scalar MSE loss, averaged over masked positions.
    """
    # 确保 pred 和 target 形状相同
    assert pred.shape == target.shape, f"pred shape {pred.shape} != target shape {target.shape}"

    # 计算逐元素平方差
    mse = (pred - target) ** 2  # [batch_size, seq_len, target_channels]

    if mask is None:
        # 无掩码时，计算所有位置的均值
        loss = mse.mean()
    else:
        # 确保 mask 与 mse 广播兼容
        if mask.dim() == 2:  # [batch_size, seq_len] -> [batch_size, seq_len, 1]
            mask = mask.unsqueeze(-1)

        # 应用掩码，计算掩码区域的损失
        masked_mse = mse * mask  # [batch_size, seq_len, target_channels]
        sum_loss = masked_mse.sum()  # 标量，掩码区域的总损失
        num_valid = mask.sum()  # 标量，掩码中 1 的数量

        # 避免除零；空 mask 时仍接到 pred 上，避免 mixed_batch 等路径下 total_loss 无 grad
        if num_valid > 0:
            loss = sum_loss / num_valid
        else:
            loss = (pred * 0).sum()

    return loss


def mae_loss(pred, target, mask=None):
    """Masked mean absolute error (L1), same masking convention as mse_loss."""
    assert pred.shape == target.shape, f"pred shape {pred.shape} != target shape {target.shape}"
    abs_diff = t.abs(pred - target)
    if mask is None:
        return abs_diff.mean()
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    masked = abs_diff * mask
    num_valid = mask.sum().clamp(min=1)
    return masked.sum() / num_valid


def huber_loss(pred, target, mask=None, delta=3.0):
    """
    Compute masked Huber loss.

    Args:
        pred (torch.Tensor): Predicted values, shape [batch_size, seq_len, target_channels] or similar.
        target (torch.Tensor): Target values, same shape as pred.
        mask (torch.Tensor, optional): Mask tensor, shape [batch_size, seq_len] or broadcastable.
                                      1 for positions to include, 0 to exclude. If None, no masking.
        delta (float, optional): Huber loss threshold. Defaults to 1.0.

    Returns:
        torch.Tensor: Scalar Huber loss, averaged over masked positions.
    """
    # Ensure pred and target have the same shape
    assert pred.shape == target.shape, f"pred shape {pred.shape} != target shape {target.shape}"

    # Compute element-wise absolute difference
    abs_diff = t.abs(pred - target)  # [batch_size, seq_len, target_channels]

    # Compute Huber loss
    quadratic = t.min(abs_diff, t.tensor(delta, device=pred.device, dtype=pred.dtype))
    linear = abs_diff - quadratic
    huber = quadratic ** 2 + delta * linear  # [batch_size, seq_len, target_channels]

    if mask is None:
        # Without mask, compute mean over all positions
        loss = huber.mean()
    else:
        # Ensure mask is broadcastable with huber
        if mask.dim() == 2:  # [batch_size, seq_len] -> [batch_size, seq_len, 1]
            mask = mask.unsqueeze(-1)

        # Apply mask and compute loss over masked positions
        masked_huber = huber * mask  # [batch_size, seq_len, target_channels]
        sum_loss = masked_huber.sum()  # Scalar, total loss over masked positions
        num_valid = mask.sum()  # Scalar, number of masked positions

        # Avoid division by zero
        if num_valid > 0:
            loss = sum_loss / num_valid
        else:
            loss = t.tensor(0.0, device=pred.device, dtype=pred.dtype)

    return loss


def smooth_loss(pred, mode='dy2'):
    """计算改进的二阶差分损失，强调方向变化"""
    # 首先检查输入是否包含 NaN/Inf
    if t.isnan(pred).any() or t.isinf(pred).any():
        warnings.warn("Found NaN/Inf in prediction tensor")
        return t.tensor(0.0, device=pred.device)
    # 计算一阶差分
    dy = pred[:, 1:] - pred[:, :-1]  # dy[i] = pred[i+1] - pred[i], 长度N-1
    # 计算二阶差分
    d2y = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]  # 长度N-2
    if mode == 'dy1':
        loss = t.mean(dy ** 2)
    else:
        loss = t.mean(d2y ** 2)

    return loss

def nll_loss(mu, log_sigma, labels, mask, alpha=1.0, sigma_regularization=0):
    """
    Calculate Negative Log-Likelihood loss with clamping and regularization.

    Args:
        mu: (batch_size, steps, bands), Gaussian distribution mean
        log_sigma: (batch_size, steps, bands), log of standard deviation
        labels: (batch_size, steps, bands), true data
        mask: (batch_size, steps, bands), 1 for valid, 0 for invalid
        alpha: Weight for NLL loss
        sigma_regularization: Weight for log_sigma regularization

    Returns:
        loss: Scalar, weighted NLL loss + regularization
    """
    # Clamp log_sigma for stability
    # log_sigma = t.clamp(log_sigma, min=-2, max=2)  # sigma in [0.135, 7.389]
    sigma = t.exp(log_sigma) + 1e-6  # Ensure positive sigma

    # Check shapes
    if mu.shape != labels.shape:
        raise ValueError(f"Expected mu shape {labels.shape}, got {mu.shape}")
    if log_sigma.shape != labels.shape:
        raise ValueError(f"Expected log_sigma shape {labels.shape}, got {log_sigma.shape}")

    # Define Gaussian distribution
    distribution = t.distributions.Normal(mu, sigma)
    likelihood = distribution.log_prob(labels)  # (batch_size, steps, bands)

    # Apply mask
    masked_likelihood = likelihood * mask
    valid_count = mask.sum() + 1e-6
    nll_loss = -t.sum(masked_likelihood) / valid_count

    # Regularization on log_sigma
    masked_log_sigma = log_sigma * mask
    reg_loss = sigma_regularization * t.sum(masked_log_sigma ** 2) / valid_count

    return alpha * nll_loss + reg_loss

def calculate_nll_per_timestep(mu, log_sigma, labels, mask):
    """
    Calculate Negative Log-Likelihood loss with clamping and regularization.

    Args:
        mu: (batch_size, steps, bands), Gaussian distribution mean
        log_sigma: (batch_size, steps, bands), log of standard deviation
        labels: (batch_size, steps, bands), true data
        mask: (batch_size, steps, bands), 1 for valid, 0 for invalid
        alpha: Weight for NLL loss
        sigma_regularization: Weight for log_sigma regularization

    Returns:
        loss: Scalar, weighted NLL loss + regularization
    """
    # Clamp log_sigma for stability
    # log_sigma = t.clamp(log_sigma, min=-2, max=2)  # sigma in [0.135, 7.389]
    sigma = t.exp(log_sigma) + 1e-6  # Ensure positive sigma

    # Check shapes
    if mu.shape != labels.shape:
        raise ValueError(f"Expected mu shape {labels.shape}, got {mu.shape}")
    if log_sigma.shape != labels.shape:
        raise ValueError(f"Expected log_sigma shape {labels.shape}, got {log_sigma.shape}")

    # Define Gaussian distribution
    distribution = t.distributions.Normal(mu, sigma)
    likelihood = distribution.log_prob(labels)  # (batch_size, steps, bands)
    masked_likelihood = likelihood * mask
    # Compute per-timestep NLL (sum over bands)
    nll_per_timestep = -masked_likelihood.sum(dim=-1, keepdim=True)  # (batch_size, steps, 1)

    return nll_per_timestep

def nll_loss_student_t(mu, log_sigma, df_raw, labels, mask, df_mode='band', df_lower=3.0, alpha=1.0, sigma_regularization=0.0001, df_regularization=0.001):
    """
    Calculate Negative Log-Likelihood loss with Student’s t-distribution, learnable df.

    Args:
        mu: (batch_size, steps, bands), Student’s t-distribution mean
        log_sigma: (batch_size, steps, bands), log of scale parameter
        df_raw:
            - per_step: (batch_size, steps, bands)
            - sequence: (batch_size, 1, 1)
            - band: (batch_size, 1, bands)
        labels: (batch_size, steps, bands), true data
        mask: (batch_size, steps, bands), 1 for valid, 0 for invalid
        df_mode: 'per_step', 'sequence', or 'band' for df sharing
        alpha: Weight for NLL loss
        sigma_regularization: Weight for log_sigma regularization
        df_regularization: Weight for df regularization

    Returns:
        loss: Scalar, weighted NLL loss + regularization
    """
    # Clamp log_sigma for stability
    # log_sigma = t.clamp(log_sigma, min=-5, max=5)  # sigma in [0.007, 148]
    sigma = t.exp(log_sigma) + 1e-6  # Ensure positive sigma

    # Check shapes
    if mu.shape != labels.shape:
        raise ValueError(f"Expected mu shape {labels.shape}, got {mu.shape}")
    if log_sigma.shape != labels.shape:
        raise ValueError(f"Expected log_sigma shape {labels.shape}, got {log_sigma.shape}")
    if mask.shape != labels.shape:
        raise ValueError(f"Expected mask shape {labels.shape}, got {mask.shape}")

    # Handle df_raw based on df_mode
    if df_mode == 'per_step':
        if df_raw.shape != labels.shape:
            raise ValueError(f"Expected df_raw shape {labels.shape}, got {df_raw.shape}")
        df = F.softplus(df_raw) + 3  # df > 3
    elif df_mode == 'sequence':
        if df_raw.shape != (labels.shape[0], 1, 1):
            raise ValueError(f"Expected df_raw shape {(labels.shape[0], 1, 1)}, got {df_raw.shape}")
        df = F.softplus(df_raw) + 3  # df > 3
        df = df.expand(-1, labels.shape[1], labels.shape[2])  # Broadcast to [batch_size, steps, bands]
    elif df_mode == 'band':
        if df_raw.shape != (labels.shape[0], 1, labels.shape[2]):
            raise ValueError(f"Expected df_raw shape {(labels.shape[0], 1, labels.shape[2])}, got {df_raw.shape}")
        df = F.softplus(df_raw) + 3  # df > 3
        df = df.expand(-1, labels.shape[1], -1)  # Broadcast to [batch_size, steps, bands]
    else:
        raise ValueError(f"Invalid df_mode: {df_mode}. Choose 'per_step', 'sequence', or 'band'.")

    # Define Student’s t-distribution
    distribution = t.distributions.StudentT(df=df, loc=mu, scale=sigma)
    likelihood = distribution.log_prob(labels)  # (batch_size, steps, bands)

    # Apply mask
    masked_likelihood = likelihood * mask
    valid_count = mask.sum() + 1e-6
    nll_loss = -t.sum(masked_likelihood) / valid_count

    # Regularization on log_sigma
    masked_log_sigma = log_sigma * mask
    reg_loss = sigma_regularization * t.sum(masked_log_sigma ** 2) / valid_count

    # Regularization on df
    masked_df = df * mask
    df_penalty = t.where(masked_df < df_lower, df_lower - masked_df, t.zeros_like(masked_df))
    df_reg_loss = df_regularization * t.sum(df_penalty) / valid_count

    return alpha * nll_loss + reg_loss + df_reg_loss

def calculate_nll_t_per_timestep(mu, log_sigma, df_raw, labels, mask, df_mode='band', df_lower=3.0):
    """
    Calculate per-timestep Negative Log-Likelihood (NLL) using Student’s t-distribution.

    Args:
        mu: (batch_size, steps, bands), Student’s t-distribution mean
        log_sigma: (batch_size, steps, bands), log of scale parameter
        df_raw:
            - per_step: (batch_size, steps, bands)
            - sequence: (batch_size, 1, 1)
            - band: (batch_size, 1, bands)
        labels: (batch_size, steps, bands), true data
        mask: (batch_size, steps, bands), 1 for valid, 0 for invalid
        df_mode: 'per_step', 'sequence', or 'band' for df sharing
        df_lower: Minimum degrees of freedom for stability

    Returns:
        nll_per_timestep: (batch_size, steps, 1), NLL summed over bands for each timestep
    """

    sigma = t.exp(log_sigma) + 1e-6  # Ensure positive sigma

    # Check shapes
    if mu.shape != labels.shape:
        raise ValueError(f"Expected mu shape {labels.shape}, got {mu.shape}")
    if log_sigma.shape != labels.shape:
        raise ValueError(f"Expected log_sigma shape {labels.shape}, got {log_sigma.shape}")
    if mask.shape != labels.shape:
        raise ValueError(f"Expected mask shape {labels.shape}, got {mask.shape}")

    # Handle df_raw based on df_mode
    if df_mode == 'per_step':
        if df_raw.shape != labels.shape:
            raise ValueError(f"Expected df_raw shape {labels.shape}, got {df_raw.shape}")
        df = F.softplus(df_raw + t.log(t.tensor(3.0, device=df_raw.device))) + 1.0  # df > 3
    elif df_mode == 'sequence':
        if df_raw.shape != (labels.shape[0], 1, 1):
            raise ValueError(f"Expected df_raw shape {(labels.shape[0], 1, 1)}, got {df_raw.shape}")
        df = F.softplus(df_raw + t.log(t.tensor(3.0, device=df_raw.device))) + 1.0  # df > 3
        df = df.expand(-1, labels.shape[1], labels.shape[2])  # Broadcast to [batch_size, steps, bands]
    elif df_mode == 'band':
        if df_raw.shape != (labels.shape[0], 1, labels.shape[2]):
            raise ValueError(f"Expected df_raw shape {(labels.shape[0], 1, labels.shape[2])}, got {df_raw.shape}")
        df = F.softplus(df_raw + t.log(t.tensor(3.0, device=df_raw.device))) + 1.0  # df > 3
        df = df.expand(-1, labels.shape[1], -1)  # Broadcast to [batch_size, steps, bands]
    else:
        raise ValueError(f"Invalid df_mode: {df_mode}. Choose 'per_step', 'sequence', or 'band'.")

    # Define Student’s t-distribution
    distribution = t.distributions.StudentT(df=df, loc=mu, scale=sigma)
    log_prob = distribution.log_prob(labels)  # (batch_size, steps, bands)

    # Apply mask
    masked_log_prob = log_prob * mask  # Zero out invalid entries

    # Compute per-timestep NLL (sum over bands)
    nll_per_timestep = -masked_log_prob.sum(dim=-1, keepdim=True)  # (batch_size, steps, 1)

    return nll_per_timestep


# def dispersive_loss(z_list, tau=0.5, lambda_list=None):
#     if lambda_list is None:
#         lambda_list = [1.0] * len(z_list)
#     assert len(lambda_list) == len(z_list), "lambda_list length must match z_list length"
#
#     total_loss = 0.0
#     for z, lam in zip(z_list, lambda_list):
#         dist = t.cdist(z, z, p=2) ** 2
#         exp_term = t.exp(-dist / tau)
#         mean_exp = t.mean(exp_term)
#         layer_loss = t.log(mean_exp)
#         total_loss += lam * layer_loss
#
#     return total_loss / len(z_list)

def dispersive_loss(z, tau=0.5, lam=0.2):
    """
    计算 InfoNCE-based Dispersive Loss（使用 L2 距离），忽略对角线元素

    参数：
        z: 编码器输出的隐空间表示，形状为 (batch, channels, steps)
        tau: 温度参数，默认为 0.5
        lam: 损失权重，默认为 1.0

    返回：
        loss: Dispersive Loss 的值
    """
    # z: (batch, channels, steps) -> (batch, channels * steps)
    z = z.view(z.shape[0], -1)  # Shape: (batch, channels * steps)
    # batch_size = z.shape[0]
    # 计算批次内样本对的平方 L2 距离
    dist = t.cdist(z, z, p=2) ** 2  # Shape: (batch, batch)

    # 计算 exp(-D/tau) 的非对角线平均值
    exp_term = t.exp(-dist / tau)  # Shape: (batch, batch)
    mean_exp = t.mean(exp_term)

    # 计算 log(mean(exp(-D/tau)))
    loss = t.log(mean_exp)

    return lam * loss

def calculate_uncertainty_per_timestep(**x):
    return None
def kl_divergence_loss(**x):
    return None

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math

# Fallback for distributed check
try:
    from dinov3.distributed import get_process_subgroup, get_subgroup_size
except ImportError:
    def get_process_subgroup(): return None
    def get_subgroup_size(): return 1

class SinkhornKnoppTeacher(nn.Module):
    """
    Sinkhorn-Knopp 算法核心模块
    针对 DDP 和 动态 Batch Size 进行了特别优化
    支持两种模式：
    1. DINO mode: 使用动态计算的 B_total（从 teacher_output.shape[0] 计算）
    2. iBOT mode: 使用传入的 n_masked_patches_tensor（更精确）
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, teacher_output, teacher_temp, n_masked_patches_tensor=None, n_iterations=3):
        """
        teacher_output: [N_local, K] 
                        注意：如果该 Rank 没有样本，这里应该传入一个 Dummy Tensor [1, K]
        teacher_temp: 温度系数
        n_masked_patches_tensor: 可选，用于 iBOT mode，指定 masked patches 的数量（tensor）
                                 如果为 None，则使用 DINO mode（从 teacher_output.shape[0] 计算）
        """
        teacher_output = teacher_output.float()
        # 与 DINOv3 一致：Q = exp(teacher/τ)；不在此处 clamp（CLS / patch 共用同一路径）
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q shape: [K, N_local]
        
        K = Q.shape[0] # Prototypes
        B_local = Q.shape[1] # Local Batch Size

        # --- 关键修复：动态计算全局 Batch Size (B) ---
        # 支持两种模式：
        # 1. iBOT mode: 使用传入的 n_masked_patches_tensor（更精确）
        # 2. DINO mode: 从 teacher_output.shape[0] 动态计算（处理不同 rank 的 batch size 差异）
        if n_masked_patches_tensor is not None:
            # iBOT mode: 使用传入的 tensor
            B = n_masked_patches_tensor.clone().float()
            if dist.is_initialized():
                dist.all_reduce(B, group=get_process_subgroup())
            B_total = B.item()
        else:
            # DINO mode: 动态计算
            # 我们不能假设 B_global = B_local * world_size，因为每个 Rank 的 valid sample 数量不同
            # 即使是 Dummy Tensor，它的 B_local (通常为1) 也会被加进去，这保证了 Sinkhorn 的分母不为0
            # 虽然这会让全局概率分布稍微稀释一点点（多了几个虚拟样本），但保证了程序不会死锁且数值稳定。
            B_tensor = torch.tensor(B_local, device=teacher_output.device, dtype=torch.float32)
            if dist.is_initialized():
                dist.all_reduce(B_tensor, group=get_process_subgroup())
            B_total = B_tensor.item()
        
        # 防止除零（极度边缘情况）
        if B_total == 0: B_total = 1.0

        # 2. 归一化矩阵总和
        # ⚠️ 与 DINOv3 源码保持一致：不使用 clamp
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q, group=get_process_subgroup())
        Q /= sum_Q

        # 3. Sinkhorn 迭代
        # ⚠️ 与 DINOv3 源码保持一致：不使用 clamp 和额外的 NaN 检查
        for _ in range(n_iterations):
            # 行归一化: 每个 Prototype 的总权重应为 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows, group=get_process_subgroup())
            Q /= sum_of_rows
            Q /= K

            # 列归一化: 每个样本的总权重应为 1/B_total
            # 注意：这是本地操作，不需要通信
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols
            Q /= B_total

        Q *= B_total  # the columns must sum to 1 so that Q is an assignment
        return Q.t() # 返回 [N_local, K]


class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.sinkhorn = SinkhornKnoppTeacher()

    def forward(self, student_logits, teacher_probs, ignore_diagonal=False):
        """
        student_logits: [n_crops_student, B, K]
        teacher_probs:  [n_crops_teacher, B, K] (已通过 Sinkhorn 处理)
        """
        student_crops, B, K = student_logits.shape
        teacher_crops, _, _ = teacher_probs.shape
        
        # ⚠️ 与 DINOv3 源码保持一致：直接使用 log_softmax，不使用 clamp
        # DINOv3 源码：student_logits = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        student_logits = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        
        if not ignore_diagonal:
            # 标准 Cross Entropy
            loss = -torch.einsum("s b k, t b k -> ", student_logits, teacher_probs)
            return loss / (B * student_crops * teacher_crops)
        else:
            # 忽略同一张图的 global-global 对应（与 DINOv3 一致）
            loss = -torch.einsum("s b k, t b k -> s t", student_logits, teacher_probs)
            min_st = min(student_crops, teacher_crops)
            loss = torch.diagonal_scatter(loss, loss.new_zeros(min_st))
            return loss.sum() / (B * student_crops * teacher_crops - B * min_st)

    def weighted_forward(self, student_logits, teacher_probs, pair_weights):
        """
        Weighted CE over student/teacher crop pairs.
        pair_weights: [n_crops_student, n_crops_teacher], zeros disable a pair.
        """
        student_crops, B, _ = student_logits.shape
        teacher_crops, _, _ = teacher_probs.shape
        log_probs = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        weights = pair_weights.to(device=log_probs.device, dtype=log_probs.dtype)
        if tuple(weights.shape) != (student_crops, teacher_crops):
            raise ValueError(
                f"pair_weights shape {tuple(weights.shape)} does not match "
                f"student/teacher crops {(student_crops, teacher_crops)}"
            )
        denom = weights.sum().clamp_min(1e-12)
        pair_loss = -torch.einsum("s b k, t b k -> s t", log_probs, teacher_probs)
        return (pair_loss * weights).sum() / (B * denom)


def cls_local_bag_loss(
    s_logits_local,
    t_probs,
    student_temp,
    lambda_contrib=0.0,
    contrib_margin=0.0,
):
    """
    Bag-style local CLS: L local views pool to one "bag" distribution vs averaged teacher targets.

    s_logits_local: [L, B, K]
    t_probs:        [N_teacher_crops, B, K] (Sinkhorn outputs; same as DINOLoss global path)
    """
    L, B, K = s_logits_local.shape
    st = float(student_temp)
    log_p_local = F.log_softmax(s_logits_local.float() / st, dim=-1)
    log_p_bag = torch.logsumexp(log_p_local - math.log(L), dim=0)
    q_avg = t_probs.mean(dim=0).detach().float()
    loss_bag = -(q_avg * log_p_bag).sum(dim=-1).mean()

    if lambda_contrib <= 0:
        return loss_bag

    p_local = F.softmax(s_logits_local.float() / st, dim=-1)
    q_dot_p = (p_local * q_avg.unsqueeze(0)).sum(dim=-1)
    q_dot_q = (q_avg * q_avg).sum(dim=-1).unsqueeze(0)
    uniform = 1.0 / float(K)
    contrib = (q_dot_p - uniform) / (q_dot_q - uniform + 1e-6)
    loss_contrib = F.relu(float(contrib_margin) - contrib).pow(2).mean()
    return loss_bag + float(lambda_contrib) * loss_contrib


def crop_view_loss(
    s_logits_local,
    t_probs,
    student_temp=0.1,
    n_crop=6,
    gamma=2.0,
    lambda_set=0.5,
    lambda_ind=1.0,
):
    """
    Set-posterior crop local CLS: pool crop views with generalized mean (gamma),
    then CE vs each teacher global (mean over teacher views after CE, not before).
    Optional per-crop anchor (loss_ind) keeps individual views tied to teacher targets.

    s_logits_local: [L, B, K], first n_crop rows are temporal crops
    t_probs:        [T, B, K]
    """
    eps = 1e-6
    nc = int(n_crop)
    s_crop = s_logits_local[:nc]
    st = float(student_temp)
    log_p_crop = F.log_softmax(s_crop.float() / st, dim=-1)

    q = t_probs.detach().float()  # [T, B, K]
    gm = float(gamma)

    # Compute the generalized-mean pooled set posterior in log-domain for stability:
    # log_score_k = (1/gamma) * log(mean_i exp(gamma * log p_i(k)))
    log_score = torch.logsumexp(gm * log_p_crop - math.log(nc), dim=0) / gm
    log_set = log_score - torch.logsumexp(log_score, dim=-1, keepdim=True)

    # L_set = (1/T) sum_t CE(q_t, p_set): pool crops first, then CE per teacher, then mean.
    # Do NOT average teacher targets before CE — that would mix nonlinear set pooling with q̄.
    loss_set = torch.tensor(0.0, device=s_crop.device, dtype=q.dtype)
    n_teacher = q.shape[0]
    for t in range(n_teacher):
        q_t = q[t]  # [B, K]
        loss_set = loss_set + -(q_t * log_set).sum(dim=-1).mean()
    loss_set = loss_set / n_teacher

    # Per-crop anchor: same rule — CE(q_t, p_i) per (crop, teacher), then average.
    loss_ind = torch.tensor(0.0, device=s_crop.device, dtype=q.dtype)
    n_terms = nc * n_teacher
    for i in range(nc):
        log_p_i = log_p_crop[i]  # [B, K]
        for t in range(n_teacher):
            loss_ind = loss_ind + -(q[t] * log_p_i).sum(dim=-1).mean()
    loss_ind = loss_ind / n_terms

    ls = float(lambda_set)
    li = float(lambda_ind)
    denom = max(ls + li, eps)
    return (ls * loss_set + li * loss_ind) / denom


class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.sinkhorn = SinkhornKnoppTeacher()

    def forward_fft_bins(self, student_logits_flat, teacher_soft_flat):
        """
        FFT proto：与 forward_masked 同式，按「行」做 CE；每行对应一个 (view, sample, freq_bin)，
        teacher_soft_flat 已为 Sinkhorn 输出。与 DINOLoss 一致：log_softmax 不做 clamp。
        """
        log_p = F.log_softmax(student_logits_flat.float() / self.student_temp, dim=-1)
        per_row = -(teacher_soft_flat.float() * log_p).sum(dim=-1)
        return per_row.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
        ibot_denom_rows=None,
    ):
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        
        # 计算 Cross Entropy
        loss = torch.sum(t.float() * F.log_softmax(s.float() / self.student_temp, dim=-1), dim=-1)
        
        if masks_weight is None:
            if student_masks_flat is not None:
                # 动态计算权重
                masks_weight = (
                    (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                    .unsqueeze(-1)
                    .expand_as(student_masks_flat)[student_masks_flat]
                )
            else:
                masks_weight = 1.0
                
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
            
        loss = loss * masks_weight

        # 分母：
        # - 若 TED 传入 ibot_denom_rows：按「可靠 patch 过滤后仍至少贡献 1 个 masked patch 的
        #   (global_student_view, batch) 行」计数，避免分子只含可靠 patch 而分母仍用未过滤 mask 行
        #   导致 iBOT 标量长期异常偏低。
        # - 否则：有「至少一个 masked patch」的 global 行数（与 DINOv3 / mask_sample_probability 语义一致）。
        if ibot_denom_rows is not None:
            denom = max(float(ibot_denom_rows), 1.0)
        elif student_masks_flat is not None:
            n_rows_active = (student_masks_flat.sum(dim=-1) > 0).sum()
            denom = n_rows_active.float().clamp(min=1.0)
        else:
            denom = 1.0

        return -loss.sum() / denom


# --- 其他辅助 Loss 保持简洁 ---

class KoleoLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.pdist = nn.PairwiseDistance(2, eps=epsilon)

    def forward(self, x):
        if x.shape[0] < 2: return torch.tensor(0.0, device=x.device)
        with torch.autocast("cuda", enabled=False):
            x = F.normalize(x, eps=self.epsilon, p=2, dim=-1)
            # 计算点积找最近邻
            dots = torch.mm(x, x.t())
            dots.view(-1)[:: (x.shape[0] + 1)].fill_(-1) # 排除自身
            _, indices = torch.max(dots, dim=1)
            distances = self.pdist(x, x[indices])
            loss = -torch.log(distances + self.epsilon).mean()
        return loss

class RobustFreqLoss(nn.Module):
    """
    FFT 重建 loss，用于序列理解（非预测）。
    根据每次输入的序列长度自适应：仅关注前 1/2 频点（低频），丢弃剩余高频。
    """
    def __init__(self, cutoff_ratio=0.5, alpha_tv=0):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        self.alpha_tv = alpha_tv

    def forward(self, rec_seq, target_seq):
        """
        rec_seq: 模型重建的序列 [B, T, C]，T 为输入长度
        target_seq: 重建目标 [B, T, C]
        cutoff 按输入长度 T 对应的频点数 n_freq 的 cutoff_ratio（默认 1/2）计算
        """
        rec_fft = torch.fft.rfft(rec_seq.float(), dim=1, norm='ortho')
        target_fft = torch.fft.rfft(target_seq.float(), dim=1, norm='ortho')
        n_freq = rec_fft.shape[1]  # 由输入序列长度 T 决定：T//2+1
        cutoff = min(n_freq, max(2, int(n_freq * self.cutoff_ratio)))
        p_low = rec_fft[:, :cutoff, :]
        t_low = target_fft[:, :cutoff, :]
        p_no_dc, t_no_dc = p_low[:, 1:, :], t_low[:, 1:, :]
        
        vec_p = torch.cat([p_no_dc.real, p_no_dc.imag], dim=-1).flatten(1)
        vec_t = torch.cat([t_no_dc.real, t_no_dc.imag], dim=-1).flatten(1)
        loss_shape = 1.0 - F.cosine_similarity(vec_p, vec_t, dim=-1).mean()
        
        loss_amp = F.l1_loss(torch.log(torch.abs(p_low) + 1e-6), torch.log(torch.abs(t_low) + 1e-6))
        loss_freq = loss_shape + 0.2 * loss_amp
        
        if self.alpha_tv > 0:
            rec_diff = rec_seq[:, 1:, :] - rec_seq[:, :-1, :]
            loss_freq += self.alpha_tv * torch.abs(rec_diff).mean()
        return loss_freq


class FFTGramAlignment(nn.Module):
    """
    频域 Gram 正则（轻约束）：
    1) 沿 patch 序列维 rFFT；
    2) 去 DC；
    3) log1p(|·|)；
    4) 乘归一化频率坐标；
    5) 频域 token L2 归一化后算 Gram，与 teacher 的 Gram 做 MSE。

    gram_mse_f_ref_bins：对 Gram MSE 乘以 min(1, (F_act/F_ref)^2)，用于 mixed_batch 变长下
    与「对 F^2 个元素取 mean」相关的梯度尺度；F_ref 取满长 seq 下去 DC 后的频点数。
    满长时 scale=1；短窗略压低；不长于 F_ref 时不额外放大（非「长比短更重要」）。None=不缩放。
    """

    def __init__(
        self,
        alpha=1.0,
        eps=1e-6,
        gram_mse_f_ref_bins: int | None = None,
        freq_keep_ratio: float = 1.0,
        freq_min_bins: int = 1,
        freq_max_bins: int = 0,
    ):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.gram_mse_f_ref_bins = gram_mse_f_ref_bins
        self.freq_keep_ratio = float(freq_keep_ratio)
        self.freq_min_bins = int(freq_min_bins)
        self.freq_max_bins = int(freq_max_bins)

    def _build_freq_tokens(self, patches):
        """
        patches: [B, N, D]，N 为 patch 数。
        returns: [B, F_no_dc, D] 或 None（长度过短时）

        rFFT 保持与 patches 相同 dtype（如 AMP 下 FP16/BF16），避免整段 [B,N,D] 先 .float()；
        幅值之后升到 FP32，再 log1p / 加权 / L2 归一化与 Gram，数值与此前全 FP32 FFT 支路一致。
        """
        x_fft = torch.fft.rfft(patches, dim=1, norm='ortho')
        x_mag = torch.log1p(torch.abs(x_fft).float())

        if x_mag.shape[1] <= 1:
            return None
        x_mag = x_mag[:, 1:, :]

        n_freq = int(x_mag.shape[1])
        if n_freq <= 1:
            freq_coord = torch.ones(
                (1, n_freq, 1),
                device=x_mag.device,
                dtype=torch.float32,
            )
        else:
            freq_coord = torch.linspace(
                0.0,
                1.0,
                steps=n_freq + 1,
                device=x_mag.device,
                dtype=torch.float32,
            )[1:].view(1, n_freq, 1)
        x_mag = x_mag * freq_coord
        x_mag = F.normalize(x_mag, p=2, dim=-1, eps=self.eps)
        return x_mag

    def _gram(self, x):
        return torch.matmul(x, x.transpose(-1, -2))

    def _resolve_keep_bins(self, f_align: int) -> int:
        if f_align <= 0:
            return 0
        ratio = float(self.freq_keep_ratio)
        if ratio <= 0:
            return 0
        if ratio >= 1.0:
            keep = int(f_align)
        else:
            keep = int(round(ratio * float(f_align)))
        keep = max(int(self.freq_min_bins), keep)
        if int(self.freq_max_bins) > 0:
            keep = min(keep, int(self.freq_max_bins))
        keep = min(keep, int(f_align))
        return max(0, keep)

    def forward(self, sPatches, tPatches):
        """
        sPatches / tPatches: [B, N, D]，teacher 在损失内部 detach。
        """
        s_freq = self._build_freq_tokens(sPatches)
        t_freq = self._build_freq_tokens(tPatches)
        if s_freq is None or t_freq is None:
            return torch.tensor(0.0, device=sPatches.device)

        f_align = min(int(s_freq.shape[1]), int(t_freq.shape[1]))
        if f_align <= 0:
            return torch.tensor(0.0, device=sPatches.device)
        keep_bins = self._resolve_keep_bins(f_align)
        if keep_bins <= 0:
            return torch.tensor(0.0, device=sPatches.device)
        s_freq = s_freq[:, :keep_bins, :]
        t_freq = t_freq[:, :keep_bins, :].detach()

        s_gram = self._gram(s_freq)
        t_gram = self._gram(t_freq)
        mse = F.mse_loss(s_gram, t_gram)
        f_act = int(s_freq.shape[1])
        if self.gram_mse_f_ref_bins is not None and int(self.gram_mse_f_ref_bins) > 0:
            f_ref = float(max(1, int(self.gram_mse_f_ref_bins)))
            # 抵消「对 F^2 个元素取 mean」带来的随 F 变化的梯度尺度；满长时 scale=1。
            # 不超过 F_ref 时不放大（避免故意让更长窗更重要）；短窗 scale<1 略压低其对总梯度的贡献。
            scale = (float(f_act) / f_ref) ** 2
            scale = min(1.0, scale)
            mse = mse * scale
        return self.alpha * mse


def fft_gram_align_masked_patch_rows(fft_mod, s_pre, t_pre, row_ids, patch_idx):
    """
    FFT Gram 仅沿「被掩码的 patch」在各自 (global view × batch 行) 内的时间顺序排列后做 rFFT-Gram。
    与 iBOT 使用同一组 pre-head patch 向量；行内掩码数 < 2 时跳过（与单点无谱维一致）。
    """
    if (
        s_pre is None
        or t_pre is None
        or row_ids is None
        or patch_idx is None
        or s_pre.shape[0] == 0
    ):
        z = torch.tensor(0.0)
        if s_pre is not None and torch.is_tensor(s_pre):
            z = z.to(device=s_pre.device, dtype=s_pre.dtype)
        return z

    device = s_pre.device
    acc = None
    n_terms = 0
    for r in torch.unique(row_ids):
        m = row_ids == r
        idx = torch.nonzero(m, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        ord_ = torch.argsort(patch_idx[idx])
        sel = idx[ord_]
        s_row = s_pre[sel].unsqueeze(0)
        t_row = t_pre[sel].unsqueeze(0)
        if s_row.shape[1] < 2:
            continue
        term = fft_mod(s_row, t_row)
        acc = term if acc is None else acc + term
        n_terms += 1
    if n_terms == 0 or acc is None:
        return torch.zeros((), device=device, dtype=s_pre.dtype)
    return acc / float(n_terms)


def temporal_neighbor_loss(z_patch):
    z1 = z_patch[:, :-1]
    z2 = z_patch[:, 1:]
    return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()


class DINOCriteria(nn.Module):
    """
    主 Loss 容器类：
    在此处处理 Dummy Data 逻辑，确保即使没有 Valid Sample 也能正常通信
    """
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        
        _st = float(getattr(args, "student_temp", 0.1))
        self.dino_loss_fn = DINOLoss(
            out_dim=args.dino_head_n_prototypes, student_temp=_st
        ).to(device)
        self.ibot_loss_fn = iBOTPatchLoss(
            patch_out_dim=args.ibot_head_n_prototypes, student_temp=_st
        ).to(device)
        f_ref_cfg = int(getattr(args, "fft_align_gram_mse_f_ref_bins", 0))
        if f_ref_cfg < 0:
            gram_f_ref = None
        elif f_ref_cfg == 0:
            pl, st = int(args.patch_len), int(args.stride)
            sq = int(getattr(args, "seq_len", 732))
            n_p = int(math.ceil((sq - pl + st) / st))
            # 与 _build_freq_tokens 一致：rfft 长度 n_p//2+1，去掉 DC 后为 n_p//2
            gram_f_ref = max(1, n_p // 2)
        else:
            gram_f_ref = max(1, f_ref_cfg)
        self.fft_gram_align_fn = FFTGramAlignment(
            alpha=1.0,
            gram_mse_f_ref_bins=gram_f_ref,
            freq_keep_ratio=float(getattr(args, "fft_align_freq_keep_ratio", 1.0)),
            freq_min_bins=int(getattr(args, "fft_align_freq_min_bins", 1)),
            freq_max_bins=int(getattr(args, "fft_align_freq_max_bins", 0)),
        ).to(device)
        self.koleo_loss_fn = KoleoLoss().to(device)

    def forward(self, outputs):
        # 解包数据
        valid_idx = outputs.get('valid_sample_indices', [])
        teacher_temp = outputs.get('teacher_temp', self.args.teacher_temp)
        lambda_weights = outputs.get('lambda_weights', {})
        
        # 权重
        l_cls = lambda_weights.get('lambda_cls_proto', self.args.lambda_cls_proto)
        l_patch = lambda_weights.get('lambda_patch_proto', self.args.lambda_patch_proto)
        l_fft_align = lambda_weights.get('lambda_fft_align', getattr(self.args, 'lambda_fft_align', 0.05))
        l_koleo = lambda_weights.get('lambda_koleo', self.args.lambda_koleo)
        l_temp = lambda_weights.get('lambda_temporal', self.args.lambda_temporal)
        l_cls_cons = lambda_weights.get('lambda_cls_cons', getattr(self.args, 'lambda_cls_cons', 0.05))

        # ----------------------------------------------------------------
        # 1. CLS Loss (DINO) - 包含防死锁逻辑
        # ----------------------------------------------------------------
        loss_cls = torch.tensor(0.0, device=self.device)
        cls_global_overlap_ratio_log = None
        cls_global_cross_weight_log = None
        cls_local_cross_weight_log = None
        cls_global_ce_log = None
        cls_short_cond_ce_log = None
        cls_short_crop_cond_ce_log = None
        cls_short_random_cond_ce_log = None
        cls_short_anchor_cond_ce_log = None
        if l_cls > 0:
            cls_data = outputs.get('cls_data', {})
            s_logits_global = cls_data.get('s_logits_global_valid') # [N, B_valid, K]
            s_logits_local = cls_data.get('s_logits_local_valid')
            t_logits_global = cls_data.get('t_logits_global_valid') # [N, B_valid, K]
            
            # 判断当前 Rank 是否有有效数据
            has_data = (s_logits_global is not None and len(valid_idx) > 0)
            
            # --- 关键：准备 Sinkhorn 输入 ---
            if has_data:
                # 正常情况：Flatten 用于 Sinkhorn
                # ⚠️ 与 DINOv3 源码保持一致：只转换为 float32，不使用额外的 NaN 检查
                t_in = t_logits_global.flatten(0, 1).detach().float() # [Total_Crops, K]
            else:
                # 异常情况：构造 Dummy 输入 [1, K]
                # 使用全0或随机数均可，Sinkhorn 会处理，最后结果会被丢弃
                dummy_k = self.args.dino_head_n_prototypes
                t_in = torch.zeros((1, dummy_k), device=self.device, dtype=torch.float32)

            # --- 执行同步 Sinkhorn ---
            # 无论是否有数据，所有 Rank 必须执行这一步
            t_out = self.dino_loss_fn.sinkhorn.forward(t_in, teacher_temp)

            # --- 计算 Loss (仅当有数据时) ---
            if has_data:
                # 还原 Shape: [N_crops, B_valid, K]
                t_probs = t_out.unflatten(0, t_logits_global.shape[:2])
                
                # Global-Global Loss
                global_loss_mode = getattr(self.args, "cls_global_loss_mode", "dino")
                if cls_data.get("cls_loss_mode", None) == "evidence_gap":
                    if bool(cls_data.get("evidence_gap_pairwise", False)):
                        teacher_row_indices = cls_data.get(
                            "evidence_gap_teacher_row_indices", None
                        )
                        if teacher_row_indices is not None:
                            teacher_row_indices = torch.as_tensor(
                                teacher_row_indices,
                                device=t_probs.device,
                                dtype=torch.long,
                            )
                            if int(teacher_row_indices.numel()) != int(s_logits_global.shape[0]):
                                raise ValueError(
                                    "evidence_gap_teacher_row_indices length must match "
                                    f"student rows, got {int(teacher_row_indices.numel())} "
                                    f"and {int(s_logits_global.shape[0])}"
                                )
                            t_probs_pair = t_probs.index_select(0, teacher_row_indices)
                        elif tuple(s_logits_global.shape) == tuple(t_probs.shape):
                            t_probs_pair = t_probs
                        else:
                            raise ValueError(
                                "evidence_gap_pairwise requires matched student/teacher "
                                "rows or evidence_gap_teacher_row_indices, got "
                                f"{tuple(s_logits_global.shape)} and {tuple(t_probs.shape)}"
                            )
                        log_p = F.log_softmax(
                            s_logits_global.float() / self.dino_loss_fn.student_temp,
                            dim=-1,
                        )
                        ce_rows = -(t_probs_pair.detach().float() * log_p).sum(dim=-1)
                        loss_g = ce_rows.mean()
                        if ce_rows.shape[0] > 0:
                            cls_global_ce_log = float(ce_rows[0].detach().mean().item())
                        if ce_rows.shape[0] > 1:
                            n_short_orig = cls_data.get("evidence_gap_n_short_original", None)
                            n_short_anchor = int(
                                cls_data.get("evidence_gap_n_short_anchor", 0) or 0
                            )
                            if n_short_orig is not None:
                                n_short_orig = int(n_short_orig)
                                short_ce_orig = ce_rows[1 : 1 + n_short_orig]
                                short_ce = short_ce_orig
                                if n_short_anchor > 0 and ce_rows.shape[0] > 1 + n_short_orig:
                                    short_ce_anchor = ce_rows[1 + n_short_orig :]
                                    cls_short_anchor_cond_ce_log = float(
                                        short_ce_anchor.detach().mean().item()
                                    )
                            else:
                                short_ce = ce_rows[1:]
                            cls_short_cond_ce_log = float(short_ce.detach().mean().item())
                            short_view_types = cls_data.get("condition_short_view_types", None)
                            if short_view_types is not None:
                                short_view_types = torch.as_tensor(
                                    short_view_types,
                                    device=ce_rows.device,
                                    dtype=torch.long,
                                ).view(-1)
                                if int(short_view_types.numel()) == int(short_ce.shape[0]):
                                    crop_mask = short_view_types == 0
                                    random_mask = short_view_types == 1
                                    if bool(crop_mask.any()):
                                        cls_short_crop_cond_ce_log = float(
                                            short_ce[crop_mask].detach().mean().item()
                                        )
                                    if bool(random_mask.any()):
                                        cls_short_random_cond_ce_log = float(
                                            short_ce[random_mask].detach().mean().item()
                                        )
                    else:
                        loss_g = self.dino_loss_fn(
                            s_logits_global, t_probs, ignore_diagonal=False
                        )
                elif (
                    global_loss_mode == "overlap_compat"
                    and int(s_logits_global.shape[0]) == 2
                    and int(t_probs.shape[0]) == 2
                ):
                    overlap_ratio = cls_data.get("cls_global_overlap_ratio", 1.0)
                    if t.is_tensor(overlap_ratio):
                        overlap_ratio = float(overlap_ratio.detach().mean().item())
                    else:
                        overlap_ratio = float(overlap_ratio)

                    min_overlap = float(
                        getattr(self.args, "cls_global_compat_min_overlap", -1.0)
                    )
                    if min_overlap < 0.0:
                        min_overlap = float(
                            getattr(self.args, "global_shift_min_overlap_ratio", 0.6)
                        )
                    min_overlap = max(0.0, min(1.0, min_overlap))

                    cross_floor = float(
                        getattr(self.args, "cls_global_compat_cross_floor", 0.25)
                    )
                    cross_floor = max(0.0, min(1.0, cross_floor))
                    self_weight = max(
                        0.0,
                        float(getattr(self.args, "cls_global_compat_self_weight", 1.0)),
                    )

                    if min_overlap >= 1.0:
                        overlap_alpha = 1.0 if overlap_ratio >= 1.0 else 0.0
                    else:
                        overlap_alpha = (overlap_ratio - min_overlap) / (1.0 - min_overlap)
                        overlap_alpha = max(0.0, min(1.0, overlap_alpha))
                    cross_weight = cross_floor + (1.0 - cross_floor) * overlap_alpha
                    cls_global_overlap_ratio_log = float(overlap_ratio)
                    cls_global_cross_weight_log = float(cross_weight)

                    pair_weights = s_logits_global.new_tensor(
                        [[self_weight, cross_weight], [cross_weight, self_weight]]
                    )
                    loss_g = self.dino_loss_fn.weighted_forward(
                        s_logits_global, t_probs, pair_weights
                    )
                else:
                    loss_g = self.dino_loss_fn(
                        s_logits_global, t_probs, ignore_diagonal=True
                    )
                
                # Local-Global Loss
                loss_l = torch.tensor(0.0, device=self.device)
                if s_logits_local is not None and s_logits_local.shape[0] > 0:
                    st_temp = self.dino_loss_fn.student_temp
                    lc = getattr(self.args, "lambda_cls_local_contrib", 0.0)
                    cm = getattr(self.args, "cls_local_contrib_margin", 0.0)
                    n_crop_cd = cls_data.get("cls_local_n_crop_views")
                    n_rand_cd = cls_data.get("cls_local_n_random_views")
                    crop_parent_ids = cls_data.get("cls_local_crop_parent_ids")
                    random_parent_ids = cls_data.get("cls_local_random_parent_ids")
                    crop_cross_overlaps = cls_data.get("cls_local_crop_cross_overlaps")
                    random_cross_overlaps = cls_data.get("cls_local_random_cross_overlaps")
                    L_loc = int(s_logits_local.shape[0])
                    local_cross_beta = max(
                        0.0,
                        float(getattr(self.args, "cls_local_cross_teacher_beta", 0.0)),
                    )
                    local_cross_normalize = bool(
                        int(getattr(self.args, "cls_local_cross_teacher_normalize", 1))
                    )
                    if local_cross_beta > 0.0:
                        _cross_log_terms = []
                        for _ovs in (crop_cross_overlaps, random_cross_overlaps):
                            if _ovs is None:
                                continue
                            if torch.is_tensor(_ovs):
                                _cross_log_terms.append(
                                    _ovs.to(device=s_logits_local.device, dtype=torch.float32).view(-1)
                                )
                            else:
                                _cross_log_terms.append(
                                    torch.as_tensor(
                                        _ovs,
                                        device=s_logits_local.device,
                                        dtype=torch.float32,
                                    ).view(-1)
                                )
                        if len(_cross_log_terms) > 0:
                            cls_local_cross_weight_log = float(
                                (
                                    local_cross_beta
                                    * torch.cat(_cross_log_terms).mean().clamp(0.0, 1.0)
                                )
                                .detach()
                                .item()
                            )

                    _loc_mode = getattr(
                        self.args, "cls_local_loss_mode", "per_view"
                    )
                    crop_gamma = float(
                        getattr(self.args, "cls_local_crop_gamma", 2.0)
                    )
                    crop_lambda_set = float(
                        getattr(self.args, "cls_local_crop_lambda_set", 0.5)
                    )
                    crop_lambda_ind = float(
                        getattr(self.args, "cls_local_crop_lambda_ind", 1.0)
                    )

                    def _per_view_local_loss(s_local, t_local=None):
                        if t_local is None:
                            t_local = t_probs
                        return self.dino_loss_fn(
                            s_local, t_local, ignore_diagonal=False
                        )

                    def _crop_loss_single_target(s_crop_group, t_local):
                        if _loc_mode == "crop_set":
                            return crop_view_loss(
                                s_crop_group,
                                t_local,
                                student_temp=st_temp,
                                n_crop=int(s_crop_group.shape[0]),
                                gamma=crop_gamma,
                                lambda_set=crop_lambda_set,
                                lambda_ind=crop_lambda_ind,
                            )
                        if _loc_mode == "bag":
                            return cls_local_bag_loss(
                                s_crop_group,
                                t_local,
                                st_temp,
                                lambda_contrib=lc,
                                contrib_margin=cm,
                            )
                        return _per_view_local_loss(s_crop_group, t_local=t_local)

                    def _parent_aware_local_loss(
                        s_local, parent_ids_local, cross_overlaps_local=None
                    ):
                        Lp = int(s_local.shape[0])
                        if (
                            parent_ids_local is not None
                            and len(parent_ids_local) == Lp
                            and t_probs.shape[0] > 0
                        ):
                            parent_ids_t = torch.as_tensor(
                                parent_ids_local, device=s_local.device, dtype=torch.long
                            )
                            cross_overlap_t = None
                            if cross_overlaps_local is not None:
                                if torch.is_tensor(cross_overlaps_local):
                                    cross_overlap_t = cross_overlaps_local.to(
                                        device=s_local.device, dtype=torch.float32
                                    ).view(-1)
                                else:
                                    cross_overlap_t = torch.as_tensor(
                                        cross_overlaps_local,
                                        device=s_local.device,
                                        dtype=torch.float32,
                                    ).view(-1)
                                if int(cross_overlap_t.numel()) != Lp:
                                    cross_overlap_t = None
                            weighted_terms = []
                            for parent_id in sorted(set(int(x) for x in parent_ids_local)):
                                if parent_id < 0 or parent_id >= int(t_probs.shape[0]):
                                    continue
                                mask = parent_ids_t == parent_id
                                if not bool(mask.any()):
                                    continue
                                s_group = s_local[mask]
                                t_group = t_probs[parent_id:parent_id + 1]
                                group_loss = _crop_loss_single_target(s_group, t_group)
                                if (
                                    local_cross_beta > 0.0
                                    and cross_overlap_t is not None
                                    and int(t_probs.shape[0]) == 2
                                ):
                                    other_id = 1 - int(parent_id)
                                    if 0 <= other_id < int(t_probs.shape[0]):
                                        cross_weight = (
                                            cross_overlap_t[mask].mean().clamp(0.0, 1.0)
                                            * local_cross_beta
                                        )
                                        cross_loss = _crop_loss_single_target(
                                            s_group, t_probs[other_id:other_id + 1]
                                        )
                                        if local_cross_normalize:
                                            group_loss = (
                                                group_loss + cross_weight * cross_loss
                                            ) / (1.0 + cross_weight)
                                        else:
                                            group_loss = group_loss + cross_weight * cross_loss
                                weighted_terms.append(
                                    (
                                        int(s_group.shape[0]) * int(t_group.shape[0]),
                                        group_loss,
                                    )
                                )
                            if len(weighted_terms) > 0:
                                denom = sum(w for w, _ in weighted_terms)
                                return sum(w * ell for w, ell in weighted_terms) / max(denom, 1)
                        return _crop_loss_single_target(s_local, t_probs)

                    def _crop_local_loss(
                        s_local,
                        n_crop_views,
                        crop_parent_ids_local=None,
                        cross_overlaps_local=None,
                    ):
                        nc = int(n_crop_views)
                        s_crop = s_local[:nc]
                        return _parent_aware_local_loss(
                            s_crop, crop_parent_ids_local, cross_overlaps_local
                        )

                    has_crop_rand_split = (
                        n_crop_cd is not None
                        and n_rand_cd is not None
                        and int(n_crop_cd) + int(n_rand_cd) == L_loc
                    )
                    use_crop_rand_split = has_crop_rand_split and _loc_mode in (
                        "crop_set",
                        "bag",
                        "per_view",
                    )

                    if use_crop_rand_split:
                        nc = int(n_crop_cd)
                        nr = int(n_rand_cd)
                        weighted_terms = []
                        if nc > 0:
                            weighted_terms.append(
                                (
                                    nc,
                                    _crop_local_loss(
                                        s_logits_local,
                                        nc,
                                        crop_parent_ids_local=crop_parent_ids,
                                        cross_overlaps_local=crop_cross_overlaps,
                                    ),
                                )
                            )
                        if nr > 0:
                            weighted_terms.append(
                                (
                                    nr,
                                    _parent_aware_local_loss(
                                        s_logits_local[nc:],
                                        random_parent_ids,
                                        random_cross_overlaps,
                                    ),
                                )
                            )
                        denom = sum(w for w, _ in weighted_terms)
                        loss_l = (
                            sum(w * ell for w, ell in weighted_terms) / denom
                            if denom > 0
                            else torch.tensor(0.0, device=self.device)
                        )
                    elif _loc_mode == "crop_set":
                        loss_l = _crop_local_loss(
                            s_logits_local,
                            L_loc,
                            crop_parent_ids_local=crop_parent_ids,
                            cross_overlaps_local=crop_cross_overlaps,
                        )
                    elif _loc_mode == "bag":
                        loss_l = _crop_local_loss(
                            s_logits_local,
                            L_loc,
                            crop_parent_ids_local=crop_parent_ids,
                            cross_overlaps_local=crop_cross_overlaps,
                        )
                    else:
                        loss_l = _per_view_local_loss(s_logits_local)
                
                # 使用 DINOv3 的 global/local scale，避免 local views 数量变化导致梯度占比漂移
                dino_global_scale = cls_data.get("dino_global_scale", None)
                dino_local_scale = cls_data.get("dino_local_scale", None)

                if dino_global_scale is None or dino_local_scale is None:
                    # 兜底：如果没有提供 scale，保持旧行为
                    loss_cls = (loss_g + loss_l) / 2.0
                else:
                    loss_cls = dino_global_scale * loss_g + dino_local_scale * loss_l

        # ----------------------------------------------------------------
        # 2. Patch Loss (iBOT) - 包含防死锁逻辑
        # ----------------------------------------------------------------
        loss_patch = torch.tensor(0.0, device=self.device)
        if l_patch > 0:
            patch_data = outputs.get('patch_data', {})
            s_masked = patch_data.get('s_patch_masked')  # [N_masked, K]
            t_masked = patch_data.get('t_patch_masked')  # [N_masked, K]

            has_patch_data = s_masked is not None and len(s_masked) > 0

            # --- 关键：准备 Sinkhorn 输入 ---
            if has_patch_data:
                # 正常情况：使用真实 teacher patch logits，并传入本 rank 的 masked 数
                t_in = t_masked.detach().float()
                n_masked_patches_tensor = torch.tensor(
                    len(s_masked), device=self.device, dtype=torch.long
                )
            else:
                # 本 rank 没有有效 masked patches：使用 dummy 向量，并传入 0，
                # 让全局 B_total 只由有数据的 rank 决定，避免数值抖动
                dummy_k = self.args.ibot_head_n_prototypes
                t_in = torch.zeros((1, dummy_k), device=self.device, dtype=torch.float32)
                n_masked_patches_tensor = torch.tensor(0, device=self.device, dtype=torch.long)

            # --- 执行同步 Sinkhorn ---
            t_out = self.ibot_loss_fn.sinkhorn.forward(
                t_in,
                teacher_temp,
                n_masked_patches_tensor=n_masked_patches_tensor,
            )

            # --- 计算 Loss ---
            if has_patch_data:
                cm = patch_data.get('collated_masks_global_valid', None)
                student_masks_flat = cm.flatten(0, 1) if cm is not None else None
                loss_patch = self.ibot_loss_fn.forward_masked(
                    student_patch_tokens_masked=s_masked,
                    teacher_patch_tokens_masked=t_out,
                    student_masks_flat=student_masks_flat,
                    n_masked_patches=len(s_masked),
                    masks_weight=patch_data.get('masks_weight_global_valid'),
                    ibot_denom_rows=patch_data.get('ibot_denom_rows'),
                )
                all_valid_tokens = patch_data.get('all_valid_tokens', None)
                if all_valid_tokens is not None:
                    # forward_masked 已按 ibot_denom_rows 做了行均值。
                    # 这里使用 denom_rows / sum(n_i)，等价于除以 mean(n_i)，
                    # 保留 n_i/m_i 缺失重权重的同时，维持与原 iBOT 标量接近的量级。
                    denom_rows = patch_data.get('ibot_denom_rows', None)
                    if denom_rows is None:
                        denom_rows = 1.0
                    if torch.is_tensor(all_valid_tokens):
                        all_valid_tokens_t = all_valid_tokens.to(
                            device=loss_patch.device, dtype=loss_patch.dtype
                        ).clamp(min=1.0)
                        denom_rows_t = torch.tensor(
                            float(denom_rows), device=loss_patch.device, dtype=loss_patch.dtype
                        )
                        loss_patch = loss_patch * (denom_rows_t / all_valid_tokens_t)
                    else:
                        loss_patch = loss_patch * (float(denom_rows) / max(float(all_valid_tokens), 1.0))

        # ----------------------------------------------------------------
        # 3. 其他 Losses (本地计算，无需 DDP 同步处理)
        # ----------------------------------------------------------------
        
        # FFT Align Loss：默认仅掩码 patch 上的谱 Gram（patch-loss 正则）；--fft_align_all_patches 时沿用全窗。
        loss_fft_align = torch.tensor(0.0, device=self.device)
        if l_fft_align > 0:
            spec_data = outputs.get('spectral_data', {})
            all_patches_mode = bool(getattr(self.args, 'fft_align_all_patches', False))
            if not all_patches_mode and l_patch > 0:
                loss_fft_align = fft_gram_align_masked_patch_rows(
                    self.fft_gram_align_fn,
                    spec_data.get('fft_masked_s_pre'),
                    spec_data.get('fft_masked_t_pre'),
                    spec_data.get('fft_masked_row_ids'),
                    spec_data.get('fft_masked_patch_idx'),
                )
            elif all_patches_mode:
                s_patch = spec_data.get('s_patch_tokens_all')
                t_patch = spec_data.get('t_patch_tokens_all')
                if s_patch is not None and t_patch is not None and len(s_patch) > 0:
                    loss_fft_align = self.fft_gram_align_fn(s_patch, t_patch)

        # Koleo Loss
        loss_koleo = torch.tensor(0.0, device=self.device)
        if l_koleo > 0:
            k_data = outputs.get('koleo_data', {})
            if k_data.get('s_z_cls_flat') is not None:
                loss_koleo = self.koleo_loss_fn(k_data['s_z_cls_flat'])

        # Temporal Loss
        loss_temporal = torch.tensor(0.0, device=self.device)
        if l_temp > 0:
            tm_data = outputs.get('temporal_data', {})
            if tm_data.get('s_z_patch_enc_valid') is not None:
                loss_temporal = temporal_neighbor_loss(tm_data['s_z_patch_enc_valid'])

        # Raw–Imputed CLS 一致性：同一样本的 raw view 与 imputed view 的 CLS 向量做余弦对齐
        loss_cls_cons = torch.tensor(0.0, device=self.device)
        if l_cls_cons > 0:
            ccd = outputs.get('cls_consistency_data')
            if ccd is not None and ccd.get('s_z_cls_per_view') is not None:
                per_view = ccd['s_z_cls_per_view']
                use_imp = ccd.get('use_imputator_per_view', [True, False])
                if (
                    per_view.dim() >= 2
                    and per_view.shape[0] >= 2
                    and per_view.shape[1] > 0
                    and len(use_imp) >= 2
                    and sum(use_imp) == 1
                ):
                    loss_cls_cons = (1.0 - F.cosine_similarity(per_view[0], per_view[1], dim=-1)).mean()

        # ----------------------------------------------------------------
        # 汇总
        # ----------------------------------------------------------------
        total_loss = (
            l_cls * loss_cls +
            l_patch * loss_patch +
            l_fft_align * loss_fft_align +
            l_koleo * loss_koleo +
            l_temp * loss_temporal +
            l_cls_cons * loss_cls_cons
        )

        # 某子 batch 上所有 gated loss 均为常数张量时，加权后仍可能 requires_grad=False，
        # 导致 scaler.backward 报错；用 student 侧张量做零乘加锚定梯度图。
        if not total_loss.requires_grad:
            for anchor in (
                outputs.get("z_global"),
                (outputs.get("spectral_data") or {}).get("s_patch_tokens_all"),
                (outputs.get("spectral_data") or {}).get("fft_masked_s_pre"),
                (outputs.get("koleo_data") or {}).get("s_z_cls_flat"),
            ):
                if anchor is not None and t.is_tensor(anchor) and anchor.requires_grad:
                    total_loss = total_loss + (anchor * 0).sum()
                    break

        loss_dict = {
            'cls': loss_cls.item(),
            'patch': loss_patch.item(),
            'fft_align': loss_fft_align.item(),
            'koleo': loss_koleo.item(),
            'temporal': loss_temporal.item(),
            'cls_cons': loss_cls_cons.item(),
        }
        if cls_global_overlap_ratio_log is not None:
            loss_dict['cls_global_overlap'] = cls_global_overlap_ratio_log
        if cls_global_cross_weight_log is not None:
            loss_dict['cls_global_cross_w'] = cls_global_cross_weight_log
        if cls_local_cross_weight_log is not None:
            loss_dict['cls_local_cross_w'] = cls_local_cross_weight_log
        if cls_global_ce_log is not None:
            loss_dict['cls_global_ce'] = cls_global_ce_log
        if cls_short_cond_ce_log is not None:
            loss_dict['cls_short_cond_ce'] = cls_short_cond_ce_log
        if cls_short_crop_cond_ce_log is not None:
            loss_dict['cls_short_crop_cond_ce'] = cls_short_crop_cond_ce_log
        if cls_short_random_cond_ce_log is not None:
            loss_dict['cls_short_random_cond_ce'] = cls_short_random_cond_ce_log
        if cls_short_anchor_cond_ce_log is not None:
            loss_dict['cls_short_anchor_cond_ce'] = cls_short_anchor_cond_ce_log
        if l_cls > 0:
            cls_data = outputs.get('cls_data', {})
            if cls_data.get("cls_loss_mode", None) == "evidence_gap":
                for src, dst in (
                    ("gap_cls_id", "gap_cls"),
                    ("gap_tokens", "gap_tokens"),
                    ("teacher_tokens", "teacher_tokens"),
                    ("student_tokens", "student_tokens"),
                    ("teacher_len", "teacher_len"),
                    ("student_len", "student_len"),
                    ("condition_ratio", "condition_ratio"),
                    ("condition_ratio_scaled_mean", "condition_ratio_scaled_mean"),
                    ("condition_ratio_scaled_min", "condition_ratio_scaled_min"),
                    ("condition_ratio_scaled_max", "condition_ratio_scaled_max"),
                    ("condition_relative_position_mean", "condition_relative_position_mean"),
                    ("condition_relative_position_min", "condition_relative_position_min"),
                    ("condition_relative_position_max", "condition_relative_position_max"),
                    ("condition_view_type_mean", "condition_view_type_mean"),
                    ("condition_crop_views", "condition_crop_views"),
                    ("condition_random_views", "condition_random_views"),
                    ("condition_anchor_views", "condition_anchor_views"),
                    ("evidence_gap_n_short_anchor", "evidence_gap_n_short_anchor"),
                    ("condition_direction_raw_norm_mean", "condition_direction_raw_norm_mean"),
                    ("condition_direction_step_norm_mean", "condition_direction_step_norm_mean"),
                    ("condition_direction_step_to_z_norm_mean", "condition_direction_step_to_z_norm_mean"),
                    ("condition_gate_mean", "condition_gate_mean"),
                    ("condition_gate_abs_mean", "condition_gate_abs_mean"),
                    ("condition_gate_raw_std", "condition_gate_raw_std"),
                    ("condition_gate_delta_norm_mean", "condition_gate_delta_norm_mean"),
                    ("condition_gate_delta_to_z_norm_mean", "condition_gate_delta_to_z_norm_mean"),
                ):
                    val = cls_data.get(src, None)
                    if val is not None:
                        loss_dict[dst] = float(val)
        
        return total_loss, loss_dict
