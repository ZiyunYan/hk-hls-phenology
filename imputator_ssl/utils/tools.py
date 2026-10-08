from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import torch as t
import torch.nn.functional as F
from utils.mcar import mcar
import math

plt.switch_backend('agg')


def fs_safe_checkpoint_dir_component(
    name: str,
    max_component_bytes: int = 220,
) -> str:
    """
    Shorten a single directory name so it fits typical NAME_MAX (255 bytes on Linux).

    Preserves uniqueness via a SHA-256 digest suffix when truncation is required.
    """
    if not isinstance(name, str):
        name = str(name)
    encoded = name.replace(os.sep, "_").replace("/", "_").encode("utf-8")
    if len(encoded) <= max_component_bytes:
        return name.replace(os.sep, "_").replace("/", "_")

    digest = hashlib.sha256(encoded).hexdigest()[:16]
    suffix = f"__h{digest}"
    suffix_b = suffix.encode("utf-8")
    budget = max_component_bytes - len(suffix_b)
    if budget < 1:
        return suffix
    prefix_b = encoded[:budget]
    while prefix_b and (prefix_b[-1] & 0xC0) == 0x80:
        prefix_b = prefix_b[:-1]
    return prefix_b.decode("utf-8") + suffix


def clamp_experiment_setting_for_checkpoint(
    setting: str,
    max_total_bytes: int = 220,
) -> str:
    """
    Ensure experiment ``setting`` (used as one path component under checkpoints/) is
    short enough for the filesystem. Keeps the trailing ``_{des}_{itr}`` segments so
    ``setting.rsplit('_', 1)`` iteration logic in ``run.py`` stays valid.
    """
    total_b = setting.encode("utf-8")
    if len(total_b) <= max_total_bytes:
        return setting
    try:
        head, _des, _itr = setting.rsplit("_", 2)
    except ValueError:
        return fs_safe_checkpoint_dir_component(setting, max_component_bytes=max_total_bytes)
    tail = setting[len(head) :]
    tail_b = tail.encode("utf-8")
    head_budget = max_total_bytes - len(tail_b)
    if head_budget < 8:
        return fs_safe_checkpoint_dir_component(setting, max_component_bytes=max_total_bytes)
    head_short = fs_safe_checkpoint_dir_component(head, max_component_bytes=head_budget)
    return head_short + tail


def resolve_fft_align_lambda_for_epoch(ep1: int, args) -> tuple[float, dict]:
    """
    Effective FFT Gram multiplier for 1-based training epoch ``ep1``.

    If ``fft_align_warmup_epochs`` > 0: linear ramp from ``fft_align_lambda_start``
    to ``lambda_fft_align`` (peak) during warmup epochs, then linear decay to
    ``fft_align_lambda_end`` by the final epoch (ignores ``fft_align_epoch_*`` gate).

    Otherwise: legacy epoch window gate, or constant ``lambda_fft_align``.
    """

    warmup = int(getattr(args, "fft_align_warmup_epochs", 0) or 0)
    peak = float(getattr(args, "lambda_fft_align", 0.0))
    train_e = max(1, int(getattr(args, "train_epochs", 1)))

    if warmup > 0:
        lam_s = float(getattr(args, "fft_align_lambda_start", 0.0))
        lam_e = float(getattr(args, "fft_align_lambda_end", 0.0))
        if ep1 <= warmup:
            if warmup <= 1:
                lam = float(peak)
            else:
                w = float(ep1 - 1) / float(warmup - 1)
                lam = lam_s + w * (peak - lam_s)
        else:
            denom = float(max(train_e - warmup, 1))
            t = float(ep1 - warmup) / denom
            t = min(1.0, max(0.0, t))
            lam = peak + t * (lam_e - peak)
        return float(lam), {
            "mode": "warmup",
            "warmup_epochs": int(warmup),
            "lambda_start": lam_s,
            "lambda_peak": float(peak),
            "lambda_end": lam_e,
        }

    start_ep = int(getattr(args, "fft_align_epoch_start", -1))
    end_raw = int(getattr(args, "fft_align_epoch_end", -1))
    active_cfg = getattr(args, "fft_align_lambda_active", None)
    base_fft = float(peak)
    active_fft = base_fft if active_cfg is None else float(active_cfg)
    inactive_fft = float(getattr(args, "fft_align_lambda_inactive", 0.0))
    if start_ep > 0 and end_raw <= 0:
        use_gate = True
        end_ep = train_e
    elif start_ep > 0 and end_raw > 0 and end_raw >= start_ep:
        use_gate = True
        end_ep = end_raw
    else:
        use_gate = False
        end_ep = end_raw
    if use_gate:
        lam = active_fft if (start_ep <= ep1 <= end_ep) else inactive_fft
    else:
        lam = base_fft
    return float(lam), {
        "mode": "gate" if use_gate else "constant",
        "use_gate": use_gate,
        "start_ep": start_ep,
        "end_ep": int(end_ep) if use_gate else end_raw,
        "active": float(active_fft),
        "inactive": float(inactive_fft),
    }


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def save_model_periodically(model, path, epoch, save_interval=5, verbose=True):
    """
    每N个epoch保存一次模型
    
    Args:
        model: 要保存的模型
        path: 保存路径（目录）
        epoch: 当前epoch（从1开始计数）
        save_interval: 保存间隔，默认每5个epoch保存一次
        verbose: 是否打印保存信息
    
    Returns:
        bool: 是否执行了保存操作
    """
    if epoch % save_interval == 0:
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        
        # 处理 DataParallel 包裹的模型
        model_to_save = model.module if hasattr(model, 'module') else model
        
        save_path = os.path.join(path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(model_to_save.state_dict(), save_path)
        
        if verbose:
            print(f'Saving model checkpoint at epoch {epoch} to {save_path}')
        return True
    return False


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def save_to_csv(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    data = pd.DataFrame({'true': true, 'preds': preds})
    data.to_csv(name, index=False, sep=',')


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_weights(weights, name='./pic/test.pdf'):
    """
    Weights visualization
    """
    fig, ax = plt.subplots()
    # im = ax.imshow(weights, cmap='plasma_r')
    im = ax.imshow(weights, cmap='YlGnBu')
    fig.colorbar(im, pad=0.03, location='top')
    plt.savefig(name, dpi=500, pad_inches=0.02)
    plt.close()


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

# def apply_mask(ori_batch_x, ori_valid_mask, p, device, mode=None, min_p=0.25):
#     if p == 0:
#         # 当 p 为 0 时，直接返回原始数据，不进行任何抹除
#         ori_batch_x = torch.nan_to_num(ori_batch_x, nan=0.0)
#         missing_mask = ori_valid_mask
#         indicating_mask = torch.zeros_like(ori_batch_x).int().to(device)
#         return ori_batch_x, ori_batch_x, missing_mask, indicating_mask

#     if mode == 'test':
#         torch.manual_seed(42)  # 或其他固定值
#         np.random.seed(42)
#     else:
#         # 训练模式下生成 [min_p, p] 范围内的均匀分布
#         p = torch.FloatTensor(1).uniform_(min_p, p).item()

#     batch_x = mcar(ori_batch_x, p)
#     missing_mask = (1 - torch.isnan(batch_x).int()).to(device)
#     indicating_mask = (ori_valid_mask - missing_mask).to(device)
#     ori_batch_x = torch.nan_to_num(ori_batch_x, nan=0.0)
#     batch_x = torch.nan_to_num(batch_x, nan=0.0)

#     return ori_batch_x, batch_x, missing_mask, indicating_mask

def apply_mask(ori_batch_x, ori_valid_mask, p, device, mode=None, min_p=0.25, use_random_p=False):
    """
    Args:
        min_p (float): 随机浮动mask ratio的下限。
        use_random_p (bool): 是否启用在 [min_p, p] 之间随机选择 mask ratio。
    """
    
    # 1. 如果 p 为 0，直接返回（无需做任何随机逻辑）
    if p == 0:
        ori_batch_x = torch.nan_to_num(ori_batch_x, nan=0.0)
        missing_mask = ori_valid_mask
        indicating_mask = torch.zeros_like(ori_batch_x).int().to(device)
        return ori_batch_x, ori_batch_x, missing_mask, indicating_mask

    # 2. 设置当前使用的 mask ratio
    current_p = p
    
    # 仅在非测试模式且启用了随机开关时，进行随机浮动
    if use_random_p and mode != 'test':
        # 确保 low <= high，防止报错
        low = min(min_p, p)
        high = max(min_p, p)
        # 在 [min_p, p] 之间均匀分布采样
        current_p = np.random.uniform(low, high)

    # 3. 处理测试模式的随机种子
    if mode == 'test':
        # 在测试/预测模式下，设置固定种子
        torch.manual_seed(42)  
        np.random.seed(42)
        # 测试模式通常强制使用固定的 p，即上面逻辑中的 current_p 保持为原始 p
        current_p = p 

    # 4. 使用 current_p 进行 mask (传入 mcar)
    batch_x = mcar(ori_batch_x, current_p)
    
    # 5. 生成对应的 mask 矩阵
    missing_mask = (1 - torch.isnan(batch_x).int()).to(device)
    indicating_mask = (ori_valid_mask - missing_mask).to(device)
    
    # 6. NaN 补 0
    ori_batch_x = torch.nan_to_num(ori_batch_x, nan=0.0)
    batch_x = torch.nan_to_num(batch_x, nan=0.0)

    return ori_batch_x, batch_x, missing_mask, indicating_mask

def apply_mask_seasons(
    ori_batch_x: torch.Tensor,
    ori_valid_mask: torch.Tensor,
    p: float,
    device: torch.device,
    mode: str = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    针对每个season进行全掩码处理，输入为已分块的数据，结合valid_mask。

    Args:
        ori_batch_x (torch.Tensor): 输入张量，形状为 [batch_size, season, steps_per_season, in_chans]
        ori_valid_mask (torch.Tensor): 有效性掩码，形状为 [batch_size, season, steps_per_season]
        p (float): 掩码概率，范围 [0, 1]
        device (torch.device): 设备
        mode (str, optional): 'test' 或 None，控制随机种子

    Returns:
        tuple: (
            ori_batch_x: 原始张量（NaN替换为0），形状为 [batch_size, season, steps_per_season, in_chans]
            batch_x: 掩码后的张量（NaN替换为0），形状为 [batch_size, season, steps_per_season, in_chans]
            missing_mask: 缺失掩码，形状为 [batch_size, season, steps_per_season]
            indicating_mask: 指示掩码，形状为 [batch_size, season, steps_per_season, in_chans]
        )
    """
    # 获取输入形状
    batch_size, season, steps_per_season, in_chans = ori_batch_x.shape
    assert ori_valid_mask.shape == (batch_size, season, steps_per_season), \
        f"Expected valid_mask shape {(batch_size, season, steps_per_season)}, but got {ori_valid_mask.shape}"

    # 如果 p == 0，直接返回原始数据，不进行掩码
    if p == 0:
        ori_batch_x = torch.nan_to_num(ori_batch_x, nan=0.0)
        missing_mask = ori_valid_mask
        indicating_mask = torch.zeros_like(ori_batch_x).int().to(device)
        return ori_batch_x, ori_batch_x, missing_mask, indicating_mask

    # 设置随机种子
    if mode == 'test':
        torch.manual_seed(42)
        np.random.seed(42)
    else:
        # 训练模式下生成 [0, p] 范围内的随机掩码概率
        p = torch.FloatTensor(1).uniform_(1e-6, p).item()

    # 生成每个season的掩码决策（伯努利分布）
    season_mask = torch.bernoulli(torch.full((batch_size, season, 1), 1 - p, device=device))  # [batch_size, season, 1]
    season_mask = season_mask.expand(-1, -1, steps_per_season)  # [batch_size, season, steps_per_season]

    # 结合valid_mask生成missing_mask
    missing_mask = ori_valid_mask * season_mask  # [batch_size, season, steps_per_season]

    # 创建batch_x，应用season级掩码
    batch_x = ori_batch_x.clone()
    mask_applied = missing_mask.unsqueeze(-1).expand(-1, -1, -1, in_chans)  # [batch_size, season, steps_per_season, in_chans]
    batch_x = batch_x.where(mask_applied == 1, torch.tensor(float('nan'), device=device))

    # 计算indicating_mask：被掩码的点（valid_mask为1但season_mask为0）
    indicating_mask = (ori_valid_mask - missing_mask).clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, in_chans).int().to(device)

    # 将NaN替换为0
    ori_batch_x = torch.nan_to_num(ori_batch_x, nan=0.0)
    batch_x = torch.nan_to_num(batch_x, nan=0.0)

    return ori_batch_x, batch_x, missing_mask, indicating_mask

def visual_results(dec_ori, dec_out, valid_mask, anomaly_mask, epoch, batch_idx, plot_anomaly=False,
                    plot_atten=None, anomalies_prob=None):  # 添加plot_atten参数
    """
    可视化原始数据、重建结果、异常检测结果以及注意力热力图
    Args:
        dec_ori: 原始数据 [B, T, N]
        dec_out: 重建结果 [B, T, N]
        valid_mask: 有效值掩码 [B, T, N]
        anomaly_mask: 异常检测掩码 [B, T, N]
        epoch: 当前训练轮次
        plot_atten: 注意力热力图矩阵 [T, N]（可选）
    """
    channels = ['Blue', 'Green','Red','NIR','SWIR1','SWIR2','NDVI']
    # 创建plots文件夹
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    # 生成保存路径
    save_path = os.path.join(plot_dir, f'epoch_{epoch}_batch_{batch_idx}.png')

    mask = anomaly_mask.detach().cpu().numpy()
    # valid_mask = valid_mask.detach().cpu().numpy()

    # 获取batch中的第一个样本
    sample_ori = dec_ori[-1]  # [T, N]
    sample_out = dec_out[-1]  # [T, N]
    sample_mask = mask[-1]  # [T, N]
    valid_sample_mask = valid_mask[-1]
    # if batch_idx == 1:
    #     print("True")
    #     save_path1 = os.path.join('plots', f'epoch_{epoch}_batch_{batch_idx}_sample_out.npy')
    #     np.save(save_path1, sample_out)
    # 计算重建误差
    reconstruction_error = np.abs(sample_ori - sample_out) * valid_sample_mask  # [T, N]

    # 创建子图 - 现在每个通道需要2行
    n_channels = sample_ori.shape[1]
    if anomalies_prob is not None:
        n_plots = n_channels * 2 + 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots))
    else:
        fig, axes = plt.subplots(n_channels * 2, 1, figsize=(15, 4 * n_channels))
    if n_channels == 1:
        axes = axes.reshape(-1)

    time_steps = np.arange(sample_ori.shape[0])

    for i in range(n_channels):
        # 原始数据和重建数据的子图
        ax1 = axes[i * 2]
        # 重建误差的子图
        ax2 = axes[i * 2 + 1]

        # 绘制原始数据和重建数据
        ax1.plot(time_steps, sample_ori[:, i], 'b-', label='Original', alpha=0.5)
        ax1.plot(time_steps, sample_out[:, i], 'r--', label='Reconstructed', alpha=0.7)
        # ax1.plot(time_steps, sample_out[:, i], 'r--', label='Season', alpha=0.7)
        if plot_anomaly:
            anomaly_points = np.where(sample_mask[:, 0] > 0.9)[0]
            if len(anomaly_points) > 0:
                ax1.scatter(anomaly_points, sample_ori[anomaly_points, i],
                            c='red', marker='x', s=100, label='Anomaly')

        ax1.set_title(f'{channels[i]} - Original vs Reconstructed')
        ax1.legend()
        ax1.grid(True)

        # 绘制重建误差
        ax2.plot(time_steps, reconstruction_error[:, i], 'g-', label='Reconstruction Error')
        # if plot_anomaly and len(anomaly_points) > 0:
        #     ax2.scatter(anomaly_points, reconstruction_error[anomaly_points, i],
        #                 c='red', marker='x', s=100, label='Anomaly')

        ax2.set_title(f'{channels[i]} - Reconstruction Error')
        ax2.legend()
        ax2.grid(True)
    # 绘制异常概率图
    if anomalies_prob is not None:
        ax_prob = axes[-1]  # 使用最后一个子图绘制异常概率
        anomalies_prob = anomalies_prob[-1].detach().cpu().numpy().flatten()  # 获取当前batch的异常概率，展平为一维
        ax_prob.plot(time_steps, anomalies_prob, 'm-', label='Anomaly Probability')
        ax_prob.set_title(f'Anomaly Probability across Time Steps')
        # ax_prob.set_ylim([0, 10])  # 异常概率的范围是 [0, 1]
        ax_prob.legend()
        ax_prob.grid(True)

    # 如果传入了注意力热力图，则绘制
    if plot_atten is not None:
        plot_atten = plot_atten[-1]
        fig_atten, ax_atten = plt.subplots(figsize=(10, 6))
        cax = ax_atten.imshow(plot_atten, aspect='auto', cmap='viridis', origin='lower')
        fig_atten.colorbar(cax, ax=ax_atten)
        ax_atten.set_title('Attention Heatmap')
        ax_atten.set_xlabel('Time Step')
        ax_atten.set_ylabel('Time Step')
        fig_atten.tight_layout()
        atten_save_path = os.path.join(plot_dir, f'epoch_{epoch}_batch_{batch_idx}_atten.png')
        plt.savefig(atten_save_path)
        plt.close(fig_atten)

    # 保存主图
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def nll_t_per_step(mu, log_sigma, df_raw, labels, mask, df_mode='band', df_lower=3.0):
    """
    Calculate Negative Log-Likelihood per time step with Student’s t-distribution for anomaly detection.

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
        nll_per_step: (batch_size, steps, bands), Negative Log-Likelihood for each time step and band
    """
    # Ensure positive sigma
    sigma = t.exp(log_sigma) + 1e-6

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
        df = F.softplus(df_raw) + 3
    elif df_mode == 'sequence':
        if df_raw.shape != (labels.shape[0], 1, 1):
            raise ValueError(f"Expected df_raw shape {(labels.shape[0], 1, 1)}, got {df_raw.shape}")
        df = F.softplus(df_raw) + 3
        df = df.expand(-1, labels.shape[1], labels.shape[2])
    elif df_mode == 'band':
        if df_raw.shape != (labels.shape[0], 1, labels.shape[2]):
            raise ValueError(f"Expected df_raw shape {(labels.shape[0], 1, labels.shape[2])}, got {df_raw.shape}")
        df = F.softplus(df_raw) + 3
        df = df.expand(-1, labels.shape[1], -1)
    else:
        raise ValueError(f"Invalid df_mode: {df_mode}. Choose 'per_step', 'sequence', or 'band'.")

    # Define Student’s t-distribution
    distribution = t.distributions.StudentT(df=df, loc=mu, scale=sigma)
    log_likelihood = distribution.log_prob(labels)  # (batch_size, steps, bands)

    # Calculate NLL (negative log-likelihood)
    nll_per_step = -log_likelihood * mask  # (batch_size, steps, bands)
    # Where mask is 0, NLL is set to 0 (invalid points)
    return nll_per_step

def z_score_detector(anomaly_prob, mask, threshold=2.5):
    """
    基于z-score的无监督异常检测函数，根据有效数据比例动态调整严格程度。

    参数:
    - anomaly_prob: (batch, steps) 异常概率，0-1之间，越大表示异常概率越高。
    - mask: (batch, steps) 有效点mask，1为有效点，0为无效点。
    - threshold: z-score的基础阈值，默认值为3.0。

    返回:
    - anomaly_label: (batch, steps) 异常标注，1为异常，0为正常。
    """
    # 只对有效点进行计算
    valid_points = anomaly_prob * mask  # 有效点的异常概率
    valid_mask = mask  # 有效点mask

    # 计算有效点的均值和标准差
    count_valid = valid_mask.sum(dim=1, keepdim=True)
    mean = (valid_points.sum(dim=1, keepdim=True) / count_valid).nan_to_num()
    std = torch.sqrt(((valid_points - mean) ** 2 * valid_mask).sum(dim=1, keepdim=True) / count_valid).nan_to_num()

    # 计算z-score
    z_scores = (valid_points - mean) / (std + 1e-8)  # 防止除零

    # 计算每个样本的有效比例
    steps = mask.size(1)
    p = count_valid / steps  # 有效点数占比

    # 动态调整阈值：当有效比例<1/3时，使用对数函数增加阈值
    adjusted_threshold = torch.where(
        p < 1/3,
        threshold + torch.log(1.0 / (3 * p + 1e-8)),  # 对数平滑调整
        threshold
    )

    # 检测异常并应用mask
    anomaly_label = (z_scores.abs() > adjusted_threshold).float() * valid_mask

    return anomaly_label.unsqueeze(-1)  # 保持输出维度一致


def process_attention(attn_outputs, register_tokens, seasons, steps_per_season=None):
    """
    处理 intra 和 inter attention，去掉 register token，归一化，并对 inter-attention 取平均。

    参数:
        attn_outputs: List of (attn_type, attn) 元组，attn_type 为 "intra" 或 "inter"，attn 为张量
        register_tokens: register token 的数量
        seasons: season 数量
        steps_per_season: 每个 season 的时间步数（可选，若不提供则自动推断）

    返回:
        inter_attn: 处理后的 inter-attention，形状为 (batch, seasons, seasons)
    """
    inter_attn = None

    for attn_type, attn in attn_outputs:
        if attn_type == "intra":
            # print('intra attn',attn.shape)
            # # 推断 batch
            # if attn.size(0) % seasons != 0:
            #     raise ValueError(f"Intra-attention 第一维 {attn.size(0)} 不能被 seasons={seasons} 整除")
            # batch = attn.size(0) // seasons
            # # 去掉 register token
            # intra_attn = attn[:, register_tokens:, register_tokens:]  # 形状: (batch * seasons, steps_per_season, steps_per_season)
            # intra_attn = torch.softmax(intra_attn, dim=-1)  # 归一化
            # intra_attn = intra_attn.view(batch, seasons, intra_attn.size(1), intra_attn.size(2))
            # print(f"处理后的 intra-attention 形状: {intra_attn.shape}")
            pass
        elif attn_type == "inter":

            # 推断 steps_per_season（如果未提供）
            if steps_per_season is None:
                # if attn.size(0) % seasons != 0:
                #     raise ValueError(f"Inter-attention 第一维 {attn.size(0)} 不能被 seasons={seasons} 整除，无法推断 steps_per_season")
                steps_per_season = attn.size(0) // seasons  # 假设 batch=1 推断 steps_per_season
            # # 验证第一维
            # if attn.size(0) % steps_per_season != 0:
            #     raise ValueError(f"Inter-attention 第一维 {attn.size(0)} 不能被 steps_per_season={steps_per_season} 整除")
            batch = attn.size(0) // steps_per_season
            # # 验证 seasons
            # if attn.size(1) != attn.size(2) or attn.size(1) != (register_tokens + seasons):
            #     raise ValueError(f"Inter-attention 形状 {attn.shape} 与 register_tokens={register_tokens} 和 seasons={seasons} 不匹配")
            # 去掉 register token
            inter_attn = attn[:, register_tokens:, register_tokens:]  # 形状: (batch * steps_per_season, seasons, seasons)
            inter_attn = torch.softmax(inter_attn, dim=-1)  # 归一化
            inter_attn = inter_attn.view(batch, steps_per_season, seasons, seasons)
            # 对 steps_per_season 维度取平均
            inter_attn = inter_attn.mean(dim=1)  # 形状: (batch, seasons, seasons)
            # print(f"处理后的 inter-attention 形状: {inter_attn.shape}")

    if inter_attn is None:
        raise ValueError("未找到 inter-attention 数据")

    return inter_attn

def restore_data(preds, dataset, batch_indices=None, window_indices=None, stride=None, seq_len=None, time_steps=None, num_pixels=None, bands=None):
    """
    还原预测结果到原始数据结构 [time_steps, bands, num_pixels]。
    
    参数：
        preds: 预测结果，形状 [total_samples, seq_len, bands]
        dataset: IterableDataset 实例，包含默认参数和原始数据
        batch_indices: 可选，自定义批次索引列表，覆盖 dataset.batch_indices
        window_indices: 可选，自定义窗口索引列表，覆盖 dataset.window_indices
        stride: 可选，自定义窗口滑动步长，覆盖 dataset.stride
        seq_len: 可选，自定义序列长度，覆盖 dataset.seq_len
        time_steps: 可选，自定义总时间步数，覆盖 dataset.time_steps
        num_pixels: 可选，自定义总像素数，覆盖 dataset.num_pixels
        bands: 可选，自定义通道数，覆盖 dataset.data_x.shape[1]
    
    返回：
        restored_preds: 还原后的预测数组，形状 [time_steps, bands, num_pixels]
    """
    # 从 dataset 获取默认参数，或使用自定义参数
    time_steps = time_steps if time_steps is not None else dataset.time_steps
    seq_len = seq_len if seq_len is not None else dataset.seq_len
    batch_size = dataset.batch_size  # batch_size 通常固定，从 dataset 获取
    num_pixels = num_pixels if num_pixels is not None else dataset.num_pixels
    bands = bands if bands is not None else dataset.data_x.shape[1]
    batch_indices = batch_indices if batch_indices is not None else dataset.batch_indices
    window_indices = window_indices if window_indices is not None else dataset.window_indices
    stride = stride if stride is not None else dataset.stride

    # 初始化输出数组
    restored_preds = np.zeros((time_steps, bands, num_pixels))
    count = np.zeros((time_steps, bands, num_pixels))  # 记录重叠次数

    # 当前样本索引
    sample_idx = 0

    # 遍历批次
    for batch_idx in batch_indices:
        start = batch_idx * batch_size
        end = min(start + batch_size, num_pixels)
        batch_pixels = list(range(start, end))
        batch_size_local = len(batch_pixels)

        # 遍历窗口
        for w_idx in window_indices:
            s_begin = w_idx * stride
            s_end = s_begin + seq_len

            if s_end > time_steps:
                continue  # 跳过无效窗口

            # 提取当前批次和窗口的预测
            batch_pred = preds[sample_idx:sample_idx + batch_size_local, :, :]  # [batch_size_local, seq_len, bands]

            # 将预测放回对应的时间步和像素位置
            for i, pixel_idx in enumerate(batch_pixels):
                restored_preds[s_begin:s_end, :, pixel_idx] += batch_pred[i, :, :]
                count[s_begin:s_end, :, pixel_idx] += 1

            sample_idx += batch_size_local

    # 处理重叠窗口：取平均值
    count[count == 0] = 1  # 防止除以零
    restored_preds /= count

    return restored_preds

def sequence2seasons(
    x: torch.Tensor, 
    season: int
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    预处理时间序列数据，根据season分割并进行零填充，自动推断seq_len和in_chans，
    同时生成填充掩码和有效性掩码（考虑NaN和padding）。

    Args:
        x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, in_chans]
        season (int): 周期数

    Returns:
        tuple: (
            处理后的张量 [batch_size, season, steps_per_season, in_chans],
            填充掩码 [season, steps_per_season] 或 None,
            有效性掩码 [batch_size, season, steps_per_season]
        )
    """
    # 从输入张量获取形状
    B, seq_len, in_chans = x.shape
    # batch_x = replace_nan_with_zero(x)
    # 生成有效性掩码，标记非NaN值为1，NaN值为0
    valid_mask = (~torch.isnan(x)).float()  # [batch_size, seq_len, in_chans]

    # 计算每个周期的步数和填充量
    steps_per_season = math.ceil(seq_len / season)
    padding = steps_per_season * season - seq_len

    # 零填充
    if padding > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, padding), mode='constant', value=0)
        # 对valid_mask也进行填充，填充部分设为0
        valid_mask = torch.nn.functional.pad(valid_mask, (0, 0, 0, padding), mode='constant', value=0)

    # 重塑张量为 [batch_size, season, steps_per_season, in_chans]
    x = x.view(B, season, steps_per_season, in_chans)
    valid_mask = valid_mask.view(B, season, steps_per_season, in_chans)

    # 生成填充掩码
    padding_mask = None
    if padding > 0:
        padding_mask = torch.ones(season, steps_per_season, device=x.device)
        padding_mask[:, -padding:] = 0

    # 合并通道维的valid_mask，确保所有通道都有效时才标记为1
    valid_mask = valid_mask.min(dim=-1)[0]  # [batch_size, season, steps_per_season]

    # 如果有填充掩码，将其应用到valid_mask
    if padding_mask is not None:
        valid_mask = valid_mask * padding_mask  # 广播：padding_mask扩展到batch_size维度

    return x, padding_mask, valid_mask


# def replace_nan_with_zero(x: torch.Tensor) -> torch.Tensor:
#     """
#     克隆输入张量并将NaN值替换为0，保持原始形状不变。

#     Args:
#         x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, in_chans]

#     Returns:
#         torch.Tensor: 克隆的张量，NaN被替换为0，形状与输入相同
#     """
#     # 克隆输入张量以避免修改原始数据
#     x_clone = x.clone()
#     # 将NaN替换为0
#     x_clone = torch.where(torch.isnan(x_clone), torch.zeros_like(x_clone), x_clone)
#     return x_clone


import torch
import torch.nn.functional as F
import math


# ==========================================
# 1. 新增：非线性时间扭曲 (核心增强)
# ==========================================
def apply_time_warp(x, warp_strength=0.2, num_control_points=5, p_warp=0.5):
    """
    [GPU向量化] 非线性时间扭曲 (Non-linear Time Warping)
    模拟物候期的非线性变化（例如：倒春寒导致前期生长缓慢，后期高温加速）
    
    原理：生成一个随机的平滑曲线作为“时间流速”，利用 grid_sample 进行重采样。
    
    Args:
        x: [B, T, C]
        warp_strength: 扭曲强度，越大时间变形越夸张 (建议 0.1-0.3)
        num_control_points: 控制点数量，越少曲线越平滑（模拟大尺度的气候变化）
    """
    B, T, C = x.shape
    device = x.device
    
    # 1. 伯努利采样：决定哪些样本需要扭曲
    do_warp = torch.bernoulli(torch.full((B,), p_warp, device=device)).bool()
    if not do_warp.any():
        return x
    
    # 2. 生成平滑的扭曲场 (Warp Field)
    # 优化：使用更快的 bilinear 插值替代 bicubic，性能提升显著
    # 我们先生成少量的随机控制点，然后插值到 T 长度，得到平滑曲线
    # noise shape: [B, 1, num_control, 1] -> 这里的维度是为了适配 interpolate 
    noise = torch.randn(B, 1, num_control_points, 1, device=device) * warp_strength
    
    # 使用 bilinear 插值生成平滑的时间偏移量 [B, 1, T, 1]（比bicubic快很多）
    # align_corners=True 保证边界对齐
    warp_field = F.interpolate(noise, size=(T, 1), mode='bilinear', align_corners=True)
    
    # 3. 构建基础网格 (Base Grid)
    # grid 的范围是 [-1, 1]，对应时间轴的 [0, T]
    # shape: [1, T, 1, 2] -> 最后一维是 (x, y) 坐标，我们只扭曲 y (时间维)
    base_grid = torch.zeros(1, T, 1, 2, device=device)
    base_grid[:, :, 0, 1] = torch.linspace(-1, 1, T, device=device) # 设置 y 坐标 (时间)
    
    # 4. 叠加扭曲
    #final_grid: [B, T, 1, 2]
    # 优化：使用 expand 而不是 clone（expand 不复制数据，只是视图）
    final_grid = base_grid.expand(B, -1, -1, -1)
    # 只对需要 warp 的样本叠加偏移量
    # warp_field 的维度是 [B, 1, T, 1]，我们需要 [B, T, 1] 加到 y 轴上
    offset = warp_field.permute(0, 2, 3, 1) # [B, T, 1, 1]
    # 优化：由于需要修改 final_grid，必须 clone（expand 是只读视图）
    final_grid = final_grid.clone()
    final_grid[do_warp, :, :, 1] += offset[do_warp, :, :, 0]
    
    # 5. 限制边界，防止采样越界
    final_grid = torch.clamp(final_grid, -1, 1)
    
    # 6. 执行重采样 (Grid Sample)
    # 优化：对于时间序列，使用更简单的线性插值可能更快
    # x 需要 reshape 成 [B, C, T, 1] 以适配 grid_sample (当作一张宽为1的图片)
    x_in = x.permute(0, 2, 1).unsqueeze(-1) # [B, C, T, 1]
    
    # sampled: [B, C, T, 1]
    # 优化：使用 'border' 替代 'reflection'，性能更好
    x_warped = F.grid_sample(x_in, final_grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    # 还原形状 [B, T, C]
    x_out = x_warped.squeeze(-1).permute(0, 2, 1)
    
    # 混合：只替换被选中的样本
    # 优化：使用 where 替代 clone + 索引赋值，可能更快
    # do_warp: [B] -> [B, 1, 1] 以匹配 [B, T, C]
    x_final = torch.where(do_warp.view(B, 1, 1), x_out, x)
    
    return x_final

# ==========================================
# 2. 新增：幅度平移 (模拟基线漂移)
# ==========================================
def apply_amplitude_shift(x, shift_range=(-0.1, 0.1), p_shift=0.5):
    """
    [GPU向量化] 随机加法平移
    模拟大气校正残差或传感器基线漂移
    """
    B, T, C = x.shape
    device = x.device
    
    # 决定哪些样本做 shift
    do_shift = torch.bernoulli(torch.full((B, 1, 1), p_shift, device=device))
    
    # 生成 shift 值 [B, 1, 1]
    low, high = shift_range
    shifts = torch.rand(B, 1, 1, device=device) * (high - low) + low
    
    return x + (shifts * do_shift)

# ==========================================
# 3. 优化：缩放与噪声 (逻辑修正)
# ==========================================
def apply_scaling_and_noise(x, sigma=0.01, scale_range=(0.95, 1.05), p_scale=0.5):
    """
    优化版本：修正运算顺序 (先缩放，后加噪)
    """
    B, T, C = x.shape
    device = x.device
    
    # 1. 缩放 (Multiplicative)
    if scale_range[0] != 1.0 or scale_range[1] != 1.0:
        do_scale = torch.bernoulli(torch.full((B, 1, 1), p_scale, device=device, dtype=x.dtype))
        low, high = scale_range
        scale_factors = torch.rand(B, 1, 1, device=device, dtype=x.dtype) * (high - low) + low
        final_scale = scale_factors * do_scale + (1.0 - do_scale)
        x = x * final_scale
    
    # 2. 噪声 (Additive) - 噪声大小不应随信号缩放
    if sigma > 0:
        noise = torch.randn_like(x) * sigma
        x = x + noise
        
    return x

# ==========================================
# 4. 优化：通道掩码 (保持不变，代码很好)
# ==========================================
def apply_channel_masking(x, mask_prob=0.3):
    if mask_prob <= 0:
        return x
    B, T, C = x.shape
    device = x.device
    
    keep_prob = 1 - mask_prob
    mask = torch.bernoulli(torch.full((B, 1, C), keep_prob, device=device, dtype=x.dtype))
    
    # 安全检查：防止所有通道全0
    channel_sums = mask.sum(dim=-1, keepdim=True)
    all_zeros = (channel_sums == 0).squeeze(-1)
    
    if all_zeros.any():
        zero_batch_indices = all_zeros.nonzero(as_tuple=False)[:, 0]
        if len(zero_batch_indices) > 0:
            rand_channels = torch.randint(0, C, (len(zero_batch_indices),), device=device)
            mask[zero_batch_indices, 0, rand_channels] = 1.0
            
    return x * mask


def apply_gaussian_noise(x, noise_std=0.01, p_noise=0.5):
    """Add Gaussian noise to the whole view."""
    if noise_std <= 0:
        return x
    if torch.rand(1, device=x.device) >= p_noise:
        return x
    noise = torch.randn_like(x) * float(noise_std)
    return x + noise

# ==========================================
# 5. 重构：Student/Teacher 输入生成逻辑
# ==========================================
_MISSING_FILL_SENTINEL = 0.0


# Compatibility no-op: missing positions are augmented after zero fill.
def _restore_missing_after_aug(x_aug, obs_valid, sentinel=_MISSING_FILL_SENTINEL):
    """Compatibility no-op; missing positions are augmented after zero fill."""
    return x_aug


def get_student_input(x_raw, aug_type='weak', is_train=True):
    """
    构建强弱视图策略
    """
    if not is_train:
        return x_raw.nan_to_num(_MISSING_FILL_SENTINEL)

    x_in = x_raw.nan_to_num(_MISSING_FILL_SENTINEL)

    if aug_type == 'none':
        return x_in

    if aug_type == 'weak':
        # Student weak: keep only light observation noise.
        return apply_gaussian_noise(
            x_in, noise_std=0.012, p_noise=0.5
        )

    if aug_type == 'weak_local':
        # Student local/random: light noise plus weak channel masking.
        x_in = apply_gaussian_noise(
            x_in, noise_std=0.015, p_noise=0.5
        )
        x_in = apply_channel_masking(x_in, mask_prob=0.12)
        return x_in

    elif aug_type == 'strong':
        # Student global: observation noise + channel masking only.
        x_in = apply_gaussian_noise(
            x_in, noise_std=0.02, p_noise=0.6
        )
        x_in = apply_channel_masking(x_in, mask_prob=0.38) 
        return x_in
        
    return x_in


def get_teacher_input(x_raw, aug_type='weak', is_train=True):
    """
    Teacher view stays numerically stable; only content differences should matter.
    """
    if not is_train:
        return x_raw.nan_to_num(_MISSING_FILL_SENTINEL)

    x_in = x_raw.nan_to_num(_MISSING_FILL_SENTINEL)
    return x_in


def imputator_sliding_window_overlap(
    x_enc,
    time_mark,
    missing_mask_orig,
    imputator,
    window_len: int,
    stride: int,
    device: torch.device,
    imp_device: torch.device,
):
    """
    超长序列：按 window_len（通常=imputator.pred_len）滑动窗口跑 imputator，步长 stride（如 244≈两年）；
    重叠区间对「缺失位置」的预测做平均；原始有效观测位置始终用 x_clean_filled，不参与平均分子。

    Args:
        x_enc: [B, T, C]，可含 nan
        time_mark: [B, T, 2] 或 None
        missing_mask_orig: [B, T] bool，True=需填补
        imputator: pred_len 通常为 window_len
        window_len: 每段送入 imputator 的时间长度（不足时在右侧重复末时刻 pad 到 window_len）
        stride: 窗口起点步长

    Returns:
        imputed_out: [B, T, C]，仅在 missing_mask_orig 为 True 且至少被一个窗口覆盖处为融合预测；
                     其余位置与 x_clean_filled 一致（由调用方再与 mask 合成 perfect target）。
    """
    B, T, C = x_enc.shape
    x_clean_filled = x_enc.nan_to_num(0.0)
    if T <= window_len:
        if imp_device != device:
            x_in_imp = x_enc.to(imp_device, non_blocking=True)
            time_imp = time_mark.to(imp_device, non_blocking=True) if time_mark is not None else None
            imputed_out = imputator(x_in_imp, time_mark=time_imp, mode="pred")
            imputed_out = imputed_out.to(device, non_blocking=True)
        else:
            imputed_out = imputator(x_enc, time_mark=time_mark, mode="pred")
        return imputed_out

    acc_sum = torch.zeros(B, T, C, device=device, dtype=x_enc.dtype)
    acc_count = torch.zeros(B, T, device=device, dtype=x_enc.dtype)

    starts = []
    s = 0
    while s < T:
        starts.append(s)
        if s + window_len >= T:
            break
        s += stride

    for start in starts:
        end = min(start + window_len, T)
        actual_len = end - start
        x_seg = x_enc[:, start:end, :]
        time_seg = time_mark[:, start:end, :] if time_mark is not None else None

        if actual_len < window_len:
            pad_len = window_len - actual_len
            last = x_seg[:, -1:, :].nan_to_num(0.0)
            x_input = torch.cat([x_seg, last.expand(B, pad_len, C)], dim=1)
            if time_seg is not None:
                last_t = time_seg[:, -1:, :]
                time_input = torch.cat([time_seg, last_t.expand(B, pad_len, time_seg.shape[-1])], dim=1)
            else:
                time_input = None
        else:
            x_input = x_seg
            time_input = time_seg

        if imp_device != device:
            xi = x_input.to(imp_device, non_blocking=True)
            ti = time_input.to(imp_device, non_blocking=True) if time_input is not None else None
            out = imputator(xi, time_mark=ti, mode="pred")
            out = out.to(device, non_blocking=True)
        else:
            out = imputator(x_input, time_mark=time_input, mode="pred")

        out = out[:, :actual_len, :]
        m = missing_mask_orig[:, start:end].float()
        acc_sum[:, start:end, :] += out * m.unsqueeze(-1)
        acc_count[:, start:end] += m

    has_pred = missing_mask_orig & (acc_count > 1e-6)
    fused = acc_sum / acc_count.unsqueeze(-1).clamp(min=1e-6)
    imputed_out = torch.where(has_pred.unsqueeze(-1), fused, x_clean_filled)
    return imputed_out


def patchify(x, patch_len, stride):
    """
    将时间序列切分为 patch tokens。

    说明：
    - 输出 token 数量为 N，其中 N = ceil((T - patch_len) / stride) + 1（当 T > patch_len）
    - 当序列长度无法整除时，会在末尾做 0-padding，确保 unfold 不会因为长度不够而报错
    - 与 unpatchify 配套使用时，unpatchify 会按 original_seq_len 截断掉 padding 部分

    Args:
        x: [B, T, C]
        patch_len: patch 长度
        stride: 步长

    Returns:
        patches: [B, N, patch_len * C]
    """
    B, T, C = x.shape

    # 计算需要的 token 数量，并推导出 unfold 需要的 padded 长度
    # N = floor((T_pad - patch_len)/stride) + 1  =>  T_pad = (N-1)*stride + patch_len
    if T <= patch_len:
        nPatches = 1
    else:
        nPatches = int(math.ceil((T - patch_len) / stride) + 1)
    tPadded = (nPatches - 1) * stride + patch_len
    padLen = max(0, tPadded - T)

    if padLen > 0:
        x = F.pad(x.permute(0, 2, 1), (0, padLen)).permute(0, 2, 1)

    x = x.permute(0, 2, 1)
    xUnfold = x.unfold(dimension=2, size=patch_len, step=stride)
    xUnfold = xUnfold.permute(0, 2, 1, 3).contiguous()
    return xUnfold.view(B, xUnfold.shape[1], -1)

def unpatchify(x_patches, original_seq_len, patch_len, c_out):
    B, N, PC = x_patches.shape
    C = c_out
    x = x_patches.view(B, N, C, patch_len)
    x = x.permute(0, 2, 1, 3).contiguous() 
    x = x.view(B, C, -1).permute(0, 2, 1)  
    if x.shape[1] > original_seq_len:
        x = x[:, :original_seq_len, :]
    return x


def random_patch_masking_patchtst_uniform(B, num_patches, mask_ratio, device):
    """
    PatchTST 论文 (Nie et al., ICLR 2023) §3.2 掩码自监督范式：
    在 patch 索引上均匀随机选取固定比例的位置置零，仅用 MSE 重建被掩码 patch。
    与 random_patch_masking_dinov3_style 返回相同三元组形状，便于在 Patch_Masked 中互换。

    Args:
        B: batch size
        num_patches: N（每样本 patch 数）
        mask_ratio: (0,1] 时约 mask_ratio * N 个 patch 被掩（与论文「40% patches」表述一致时用 0.4）
        device: torch device

    Returns:
        collated_masks: [B, N] bool，True = 该 patch 被 SSL 掩码
        mask_indices_list: flatten 的 True 位置索引（与 dinov3 版一致）
        masks_weight: 按行归一的权重（与 dinov3 版一致）
    """
    N = int(num_patches)
    if N <= 0:
        collated_masks = torch.zeros(B, 0, dtype=torch.bool, device=device)
        return collated_masks, torch.empty(0, dtype=torch.long, device=device), torch.empty(0, device=device)

    ratio = float(mask_ratio)
    ratio = max(0.0, min(1.0, ratio))
    n_masked = int(round(N * ratio))
    n_masked = max(0, min(N, n_masked))

    if n_masked == 0:
        collated_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
    else:
        noise = torch.rand(B, N, device=device)
        perm = noise.argsort(dim=-1)
        idx = perm[:, :n_masked]
        collated_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
        collated_masks.scatter_(1, idx, True)

    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    return collated_masks, mask_indices_list, masks_weight


def random_patch_masking_dinov3_style(
    B,
    mask_ratio_tuple,
    mask_sample_probability,
    num_patches,
    device,
    block_ratio: float = 0.8,
    block_size: int = 5,
):
    """
    模仿DINOv3的mask生成策略（优化版本：避免CPU-GPU同步）
    Args:
        B: batch size
        mask_ratio_tuple: (min_ratio, max_ratio)，例如(0.1, 0.5)
            mask_sample_probability: 在 B 行中约有多少比例会生成非空 patch mask（0~1），例如 0.5 表示约一半行无 mask token（DINO 风格）
        num_patches: patch数量
        device: device
    Returns:
        masks: [B, num_patches] bool tensor, True表示masked
        mask_indices_list: flatten的mask索引列表
        masks_weight: 每个masked patch的权重
    """
    N = num_patches
    # 期望有 mask 的行数：四舍五入到 [0, B]，避免 int(B*p) 在 B 较小时恒为 0（例如 B=1, p=0.5）
    p = float(mask_sample_probability)
    p = max(0.0, min(1.0, p))
    n_samples_masked = max(0, min(B, int(B * p + 0.5)))
    
    # 生成mask ratio的线性分布
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1, device=device)
    
    # 优化：直接在GPU上生成所有mask，避免循环和CPU-GPU同步
    masks_tensor = torch.zeros(B, N, dtype=torch.bool, device=device)
    
    # 为需要mask的样本生成mask
    # 优化：避免在循环中调用 .item()，直接在GPU上计算
    for i in range(n_samples_masked):
        prob_max = probs[i + 1]  # 保持在GPU上，不调用.item()
        n_masked = int((N * prob_max).item())  # 只在需要整数时调用.item()
        # 生成mask（Block + Random混合策略）
        mask = generate_single_mask(N, n_masked, device, block_ratio=block_ratio, block_size=block_size)
        masks_tensor[i] = mask
    
    # 未mask的样本保持为全0（已经是False）
    
    # 优化：使用GPU上的索引打乱，避免CPU-GPU同步
    indices = torch.randperm(B, device=device)
    collated_masks = masks_tensor[indices]  # [B, N]
    
    # 生成mask_indices_list（flatten的索引）
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    
    # 生成masks_weight（用于平衡不同样本的mask数量）
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    
    return collated_masks, mask_indices_list, masks_weight


def generate_single_mask(N, n_masked, device, block_ratio: float = 0.8, block_size: int = 5):
    """
    为单个样本生成mask（Block + Random混合策略，优化版本）
    Args:
        N: patch数量
        n_masked: 需要mask的patch数量
    Returns:
        mask: [N] bool tensor
    """
    if n_masked == 0:
        return torch.zeros(N, dtype=torch.bool, device=device)
    
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    
    # Block mask: block_ratio 的 mask 是连片的
    target_block_mask = int(n_masked * block_ratio)
    
    # 优化：减少循环，使用向量化操作
    # 生成block mask
    num_blocks = int(math.ceil(target_block_mask / block_size * 1.2))
    if num_blocks > 0 and N >= block_size:
        # 优化：批量生成block mask，使用向量化操作替代循环
        rand_starts = torch.randint(0, max(1, N - block_size + 1), (num_blocks,), device=device)
        # 优化：向量化设置mask，使用高级索引一次性设置所有 blocks
        # 为每个 block 生成索引范围
        block_indices = rand_starts.unsqueeze(1) + torch.arange(block_size, device=device).unsqueeze(0)  # [num_blocks, block_size]
        block_indices = block_indices.clamp(max=N-1)  # 限制边界
        block_indices = block_indices.flatten()  # [num_blocks * block_size]
        # 使用高级索引一次性设置所有 mask（去重以避免重复设置）
        unique_indices = torch.unique(block_indices)
        mask[unique_indices] = True
    
    # 补齐random mask
    # 优化：只在必要时调用 .item()，减少CPU-GPU同步
    current_masked = mask.sum().item()
    to_fill = n_masked - current_masked
    if to_fill > 0:
        available_indices = (~mask).nonzero().flatten()
        if len(available_indices) > 0:
            n_to_select = min(to_fill, len(available_indices))
            selected = available_indices[torch.randperm(len(available_indices), device=device)[:n_to_select]]
            mask[selected] = True
    
    return mask


def random_patch_masking(B, mask_rate, device, num_patches):
    """
    [混合策略] Mixed Masking: Block + Random
    全向量化实现，无 CPU 循环
    1 (True) = 被遮挡 (Masked) -> 这是 Student 看不到的，需要预测的部分。
    0 (False) = 可见 (Visible) -> 这是 Student 能看到的，作为上下文的部分。
    """
    N = num_patches
    
    # 参数配置
    block_size = 5     # 块大小
    block_ratio = 0.8  # 80% 的掩码是连片的
    
    # 1. 计算 Block Mask 的目标数量
    target_mask_total = int(N * mask_rate)
    target_block_mask = int(target_mask_total * block_ratio)
    
    # 2. 生成 Block Mask (连片)
    num_blocks = int(math.ceil(target_block_mask / block_size * 1.2)) 
    rand_starts = torch.randint(0, max(1, N - block_size + 1), (B, num_blocks), device=device)
    offsets = torch.arange(block_size, device=device).view(1, 1, -1)
    block_indices = rand_starts.unsqueeze(-1) + offsets
    block_indices = block_indices.view(B, -1)
    
    # 3. 生成 Mask 矩阵
    mask = torch.zeros((B, N), device=device)
    src = torch.ones_like(block_indices, dtype=torch.float)
    mask.scatter_(1, block_indices, src) # 填入 Block
    
    # 4. 补齐 Random Mask (随机散点)
    current_masked_count = mask.sum(dim=-1) # [B]
    to_fill = target_mask_total - current_masked_count
    to_fill = to_fill.clamp(min=0).long()
    
    noise = torch.rand((B, N), device=device)
    noise.masked_fill_(mask.bool(), 1e9) # 已经 Mask 的不要再选

    max_fill = to_fill.max().item()
    if max_fill > 0:
        _, indices = torch.topk(noise, k=max_fill, dim=-1, largest=False)
        
        # 生成 mask 指示哪些索引有效
        batch_range = torch.arange(max_fill, device=device).unsqueeze(0)
        valid_indices_mask = batch_range < to_fill.unsqueeze(1) # [B, max_fill]
        
        src_fill = valid_indices_mask.float()
        mask.scatter_(1, indices, src_fill)

    mask = mask.clamp(0, 1)
    return mask

def create_smooth_target(x_filled, missing_mask_bool, kernel_size=5):
    """
    [适配版: 1=无效, 0=有效]
    使用 AvgPool1d 近似线性插值来填补 0 值
    """
    # x_filled: [B, T, C] (已填0或原始值)
    # missing_mask_bool: [B, T] (True/1=缺失/无效, False/0=有效)
    
    B, T, C = x_filled.shape
    
    # 1. 准备数据
    x = x_filled.permute(0, 2, 1) # [B, C, T]
    
    # 将 bool 掩码转为 float: 1.0=无效, 0.0=有效
    mask_invalid = missing_mask_bool.float().unsqueeze(1)  # [B, 1, T]
    mask_valid = 1.0 - mask_invalid
    x_masked = x * mask_valid  # 广播乘法

    # 2. 执行平滑（优化：合并操作）
    padding = kernel_size // 2
    x_smooth = F.avg_pool1d(x_masked, kernel_size, stride=1, padding=padding, count_include_pad=False)
    weight_smooth = F.avg_pool1d(mask_valid, kernel_size, stride=1, padding=padding, count_include_pad=False)
    
    # 3. 插值（优化：合并计算）
    x_interpolated = x_smooth / (weight_smooth + 1e-6)
    
    # 4. 融合（优化：使用where替代expand+乘法）
    mask_invalid_expanded = mask_invalid.expand(-1, C, -1)  # [B, C, T]
    x_final = torch.where(mask_invalid_expanded, x_interpolated, x)

    return x_final.permute(0, 2, 1)  # [B, T, C]


def generate_local_view_crop(x, target_len, device):
    """
    通过crop生成局部视图（向量化版本）
    Args:
        x: [B, T, C] 原始序列
        target_len: 目标长度（时间步数）
        device: device
    Returns:
        x_crop: [B, target_len, C]
        start_indices: [B] 每个样本的起始索引
    """
    B, T, C = x.shape
    max_start = max(0, T - target_len)
    start_indices = torch.randint(0, max_start + 1, (B,), device=device)
    
    # 向量化实现：使用gather或高级索引
    # 创建索引矩阵 [B, target_len]
    batch_indices = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
    time_indices = start_indices.unsqueeze(1) + torch.arange(target_len, device=device).unsqueeze(0)  # [B, target_len]
    
    # 使用gather或高级索引提取
    x_crop = x[batch_indices, time_indices]  # [B, target_len, C]
    return x_crop, start_indices


def generate_local_view_random_sample(x, num_tokens, device):
    """
    通过随机抽取token生成局部视图（向量化版本）
    Args:
        x: [B, T, C] 原始序列（已经patchify后的token序列，T是token数量）
        num_tokens: 要抽取的token数量
        device: device
    Returns:
        x_sampled: [B, num_tokens, C]
        token_indices: [B, num_tokens] 抽取的token索引
    """
    B, T, C = x.shape
    
    # 向量化实现：为每个batch生成随机索引
    # 生成随机数矩阵 [B, T]，然后取topk
    rand_vals = torch.rand(B, T, device=device)
    _, token_indices = torch.topk(rand_vals, k=num_tokens, dim=1)  # [B, num_tokens]
    token_indices, _ = torch.sort(token_indices, dim=1)  # 排序以保持时间顺序
    
    # 使用gather提取
    batch_indices = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
    x_sampled = x[batch_indices, token_indices]  # [B, num_tokens, C]
    
    return x_sampled, token_indices
