import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
from layers.Embedding import Embedding
from utils.losses import smooth_loss, mse_loss, huber_loss, mae_loss
from utils.tools import apply_mask
import os


def _lon_lat_fourier_features(lon_lat_deg: torch.Tensor, n_freqs: int) -> torch.Tensor:
    """多频 sin/cos 经纬度特征，与 backbone.lon_lat_fourier_features 逻辑一致。"""
    lon = lon_lat_deg[:, 0:1] * (math.pi / 180.0)
    lat = lon_lat_deg[:, 1:2] * (math.pi / 180.0)
    parts = []
    for i in range(n_freqs):
        w = 2.0 ** i
        parts.extend([torch.sin(w * lon), torch.cos(w * lon),
                       torch.sin(w * lat), torch.cos(w * lat)])
    return torch.cat(parts, dim=-1)


class Imputator(nn.Module):
    """
    仅用于填补：Encoder 前可选拼接 storage（register）tokens，缓和注意力异常；无 CLS，无 encode 接口。
    输出头只作用于时间步 token。
    """
    def __init__(self, configs):
        super(Imputator, self).__init__()
        self.pred_len = int(getattr(configs, "seq_len", None) or getattr(configs, "label_len", 366))
        self.output_attention = True
        self.d_model = getattr(configs, 'd_model', 128)
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.e_layers = getattr(configs, 'e_layers', 4)
        self.d_ff = getattr(configs, 'd_ff', 256)
        # imp_n_storage_tokens 默认 2；-1 时回退到 n_storage_tokens（>0）否则 2
        _ims = int(getattr(configs, 'imp_n_storage_tokens', 2))
        if _ims >= 0:
            self.n_storage_tokens = max(0, _ims)
        else:
            _ns = int(getattr(configs, 'n_storage_tokens', 0))
            self.n_storage_tokens = _ns if _ns > 0 else 2

        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, self.n_storage_tokens, self.d_model))
            nn.init.normal_(self.storage_tokens, std=0.02)
        else:
            self.storage_tokens = None

        # for the 1st block
        self.embedding = Embedding(
            d_in=configs.enc_in + 3,
            d_model=self.d_model,
            with_pos=True,
            embed_type=configs.embed,
            freq=configs.freq,
            n_max_steps=self.pred_len,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, diag_mask_flag=False), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.output_projection = nn.Linear(self.d_model, configs.enc_in, bias=True)

        self._lon_lat_n_freqs = int(getattr(configs, 'lon_lat_n_fourier_freqs', 4))
        use_ll = bool(getattr(configs, 'use_lon_lat_embed', 1))
        if use_ll:
            self.lon_lat_proj = nn.Linear(4 * self._lon_lat_n_freqs, self.d_model)
        else:
            self.lon_lat_proj = None

    @property
    def _prefix_len(self):
        return self.n_storage_tokens

    def forward(self, x_enc, mask=None, time_mark=None, lon_lat_emb=None):
        enc_out = self.embedding(x_enc, mask.unsqueeze(-1), time_mark)
        if lon_lat_emb is not None:
            enc_out = enc_out + lon_lat_emb
        if self.n_storage_tokens > 0:
            B = enc_out.shape[0]
            st = self.storage_tokens.expand(B, -1, -1)
            enc_out = torch.cat([st, enc_out], dim=1)
        enc_out, _ = self.encoder(enc_out)
        patch_h = enc_out[:, self._prefix_len :, :]
        dec_out = self.output_projection(patch_h)
        return dec_out

def cal_rec_loss(
    pred,
    target,
    mask,
    missing_mask,
    alpha,
    beta,
    missing_loss_weight=0.0,
    huber_delta=1.0,
    rec_loss_type='mse',
    smooth_mode='dy2',
    trim_topk_per_seq=0,
    trim_min_keep=8,
):
    """重建项 + 平滑项。rec_loss_type: mse | huber | mae；smooth_mode: dy1 | dy2（见 utils.losses.smooth_loss）"""
    def _apply_topk_trim_mask(predTensor, targetTensor, maskTensor, topkPerSeq, minKeep):
        """
        在每个样本内，按“时间点”忽略高误差监督：
        - 先把每个时间点各通道误差聚合为一个分数；
        - 再选分数最高的若干时间点；
        - 被选中的时间点会联动忽略该时间点下所有可监督通道（通道间相关）。
        仅修改重建项监督掩码，不影响平滑项。
        """
        try:
            if topkPerSeq <= 0:
                return maskTensor

            trimmedMask = maskTensor.clone()
            batchSize = predTensor.shape[0]

            for batchIdx in range(batchSize):
                sampleMaskBool = (trimmedMask[batchIdx] > 0)  # [T, C]
                supervisedPerTime = sampleMaskBool.sum(dim=-1)  # [T]
                validTimeMask = supervisedPerTime > 0
                validTimeCount = int(validTimeMask.sum().item())
                if validTimeCount <= 0:
                    continue

                totalSupervisedPoints = int(supervisedPerTime.sum().item())
                maxDroppablePoints = max(0, totalSupervisedPoints - int(minKeep))
                if maxDroppablePoints <= 0:
                    continue

                # 时间点误差分数：对该时间点下“可监督通道”做均值，避免通道数影响排序
                absErr = (predTensor[batchIdx] - targetTensor[batchIdx]).abs()  # [T, C]
                supervisedPerTimeFloat = supervisedPerTime.clamp_min(1).float()
                timeErrScore = (absErr * sampleMaskBool.float()).sum(dim=-1) / supervisedPerTimeFloat  # [T]

                candidateTimeIdx = torch.nonzero(validTimeMask, as_tuple=False).squeeze(-1)
                if candidateTimeIdx.numel() == 0:
                    continue
                candidateScores = timeErrScore[candidateTimeIdx]
                _, sortPos = torch.sort(candidateScores, descending=True)
                sortedTimeIdx = candidateTimeIdx[sortPos]

                # 选 topK 个高误差时间点，同时保证至少保留 minKeep 个监督点
                selectedTimeIdx = []
                droppedPoints = 0
                maxTopk = min(int(topkPerSeq), int(sortedTimeIdx.numel()))
                for timeIdx in sortedTimeIdx[:maxTopk]:
                    pointsAtTime = int(supervisedPerTime[timeIdx].item())
                    if pointsAtTime <= 0:
                        continue
                    if droppedPoints + pointsAtTime > maxDroppablePoints:
                        continue
                    selectedTimeIdx.append(int(timeIdx.item()))
                    droppedPoints += pointsAtTime

                if not selectedTimeIdx:
                    continue

                trimmedMask[batchIdx, selectedTimeIdx, :] = 0.0

            return trimmedMask
        except Exception:
            return maskTensor

    mask = _apply_topk_trim_mask(pred, target, mask, trim_topk_per_seq, trim_min_keep)

    if rec_loss_type == 'mse':
        rec_loss = mse_loss(pred, target, mask)
    elif rec_loss_type == 'huber':
        rec_loss = huber_loss(pred, target, mask, delta=huber_delta)
    elif rec_loss_type == 'mae':
        rec_loss = mae_loss(pred, target, mask)
    else:
        raise ValueError(f"cal_rec_loss: unknown rec_loss_type={rec_loss_type!r}, use mse|huber|mae")

    smo_loss = smooth_loss(pred, mode=smooth_mode)
    return alpha * rec_loss + beta * smo_loss


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.imputation_model = Imputator(configs)
        self.configs = configs
        self.dropout = nn.Dropout(p=0.2)
        self.mask_rate = configs.mask_rate
        self.geo_dropout_p = float(getattr(configs, 'geo_dropout_p', 0.5))
        self.imp_rec_loss = str(getattr(configs, 'imp_rec_loss', 'mse')).lower()
        self.imp_huber_delta = float(getattr(configs, 'imp_huber_delta', 1.0))
        self.imp_rec_alpha = float(getattr(configs, 'imp_rec_alpha', 1.0))
        self.imp_smooth_beta = float(getattr(configs, 'imp_smooth_beta', 0.5))
        self.imp_smooth_mode = str(getattr(configs, 'imp_smooth_mode', 'dy2')).lower()
        if self.imp_smooth_mode not in ('dy1', 'dy2'):
            self.imp_smooth_mode = 'dy2'
        self.impTrimTopkPerSeq = int(getattr(configs, 'imp_trim_topk_per_seq', 0))
        self.impTrimMinKeep = int(getattr(configs, 'imp_trim_min_keep', 8))
        # 只在主进程输出（避免DDP重复输出）
        try:
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                print('mask ratio:', self.mask_rate)
                print(
                    f'imputator loss: rec={self.imp_rec_loss} (huber_delta={self.imp_huber_delta}) '
                    f'alpha={self.imp_rec_alpha} smooth_beta={self.imp_smooth_beta} smooth_mode={self.imp_smooth_mode}; '
                    f'n_storage_tokens={self.imputation_model.n_storage_tokens}'
                )
                print(
                    f'imputator topk trim: topk_per_seq={self.impTrimTopkPerSeq} '
                    f'min_keep={self.impTrimMinKeep}'
                )
        except Exception:
            print('mask ratio:', self.mask_rate)
            print(
                f'imputator loss: rec={self.imp_rec_loss} (huber_delta={self.imp_huber_delta}) '
                f'alpha={self.imp_rec_alpha} smooth_beta={self.imp_smooth_beta} smooth_mode={self.imp_smooth_mode}; '
                f'n_storage_tokens={self.imputation_model.n_storage_tokens}'
            )
            print(
                f'imputator topk trim: topk_per_seq={self.impTrimTopkPerSeq} '
                f'min_keep={self.impTrimMinKeep}'
            )

    def _compute_lon_lat_emb(self, lon_lat, B, T, device, dtype, training_drop=True):
        """计算 lon_lat Fourier 嵌入并施加 geo dropout；返回 [B,T,d_model] 或 None。"""
        proj = self.imputation_model.lon_lat_proj
        if lon_lat is None or proj is None:
            return None
        ll_sample = lon_lat[:, 0, :] if lon_lat.dim() == 3 else lon_lat  # [B,2]
        nan_mask = torch.isnan(ll_sample).any(dim=-1, keepdim=True)  # [B,1]
        ll_clean = ll_sample.nan_to_num(0.0)
        feat = _lon_lat_fourier_features(ll_clean, self.imputation_model._lon_lat_n_freqs)  # [B,4*nf]
        emb = proj(feat)  # [B, d_model]
        emb = emb.masked_fill(nan_mask, 0.0)
        if self.training and training_drop and self.geo_dropout_p > 0:
            keep = (torch.rand(B, 1, device=device) >= self.geo_dropout_p).to(dtype)
            emb = emb * keep
        return emb.unsqueeze(1).expand(-1, T, -1)  # [B,T,d_model]

    def _prepare_forecast_label(self, next_x):
        """Prepare input tensor with padding and masking."""

        next_valid_mask = (1 - torch.isnan(next_x).int()).to(next_x.device)

        # Apply mask
        next_batch, _, _, _ = apply_mask(
            next_x, next_valid_mask, 0, next_x.device
        )

        return next_batch, next_valid_mask

    def _prepare_input(self, x_enc, mask_ratio, valid_mask=None, disable_random_mask=False):
        """Prepare input tensor with padding and masking."""
        # batch_size, seq_len, bands = x_enc.shape
        if valid_mask is None:
            valid_mask = (1 - torch.isnan(x_enc).int()).to(x_enc.device)

        # 预测/推理场景：强制不对有效点做额外随机掩码，仅保留原始缺失。
        if disable_random_mask:
            batch_x, batch_x_masked, missing_mask, indicating_mask = apply_mask(
                x_enc, valid_mask, 0.0, x_enc.device,
                mode='test', min_p=0.0, use_random_p=False,
            )
            return batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask

        # Apply mask：随机掩码率在 [imp_mask_min_p, max(mask_rate, min_p)]（默认 min_p=0.4）
        min_p = float(getattr(self.configs, 'imp_mask_min_p', 0.4))
        p_upper = max(float(mask_ratio), min_p)
        batch_x, batch_x_masked, missing_mask, indicating_mask = apply_mask(
            x_enc, valid_mask, p_upper, x_enc.device,
            mode=None, min_p=min_p, use_random_p=True,
        )

        return batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask

    def forward(self, x_enc, time_mark=None, valid_mask=None, next_x_enc=None, mode='train',
                imputator=None, mask_rate_v1=None, mask_rate_v2=None, teacher_temp=None,
                iteration=None, total_iterations=None, current_epoch=None, lon_lat=None):
        batch_size, seq_len, bands = x_enc.shape
        if valid_mask is None:
            valid_mask = (1 - torch.isnan(x_enc).int()).to(x_enc.device)

        if mode == 'train':
            batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
                x_enc, self.mask_rate, valid_mask
            )
            recMask = indicating_mask.float()
            ll_emb = self._compute_lon_lat_emb(lon_lat, batch_size, seq_len, x_enc.device, x_enc.dtype, training_drop=True)
            dec_out = self.imputation_model(batch_x_masked, mask=(1-missing_mask[:, :, 0]).float(), time_mark=time_mark, lon_lat_emb=ll_emb)

            loss = cal_rec_loss(
                dec_out,
                batch_x,
                recMask,
                missing_mask,
                alpha=self.imp_rec_alpha,
                beta=self.imp_smooth_beta,
                missing_loss_weight=0.05,
                huber_delta=self.imp_huber_delta,
                rec_loss_type=self.imp_rec_loss,
                smooth_mode=self.imp_smooth_mode,
                trim_topk_per_seq=self.impTrimTopkPerSeq,
                trim_min_keep=self.impTrimMinKeep,
            )
            return {'ssl_loss': loss, 'log_vars': {'rec_loss': loss.detach()}}

        elif mode == 'fine-tune':
             pass

        elif mode == 'pred' or mode == 'test':
            batch_x, batch_x_masked, valid_mask, missing_mask, indicating_mask = self._prepare_input(
                x_enc, 0, valid_mask, disable_random_mask=True
            )
            time_mark = self.dropout(time_mark) if time_mark is not None else None
            ll_emb = self._compute_lon_lat_emb(lon_lat, batch_size, seq_len, x_enc.device, x_enc.dtype, training_drop=False)
            dec_out = self.imputation_model(batch_x_masked, mask=(1-missing_mask[:, :, 0]).float(), time_mark=time_mark, lon_lat_emb=ll_emb)

            return dec_out

        else:
            raise ValueError(f"Supported mode: train, fine-tune, or pred")


def load_pretrained_imputator(configs, checkpoint_path, target_device=None, use_ddp=False, local_rank=None):
    """
    加载预训练 Imputator (基于 Model 包装类)，支持 DDP 模式分散显存占用。
    
    Args:
        configs: 模型配置 (需包含 devices='1,2,3' 或 device_ids=[1,2,3])
        checkpoint_path: 权重路径
        target_device: 目标设备，如果为None则自动选择
                      - 'cuda:X': 加载到指定GPU
                      - None: 自动选择（DDP模式下使用当前进程的GPU）
        use_ddp: 是否使用 DDP 模式（每个进程加载到自己的GPU）
        local_rank: DDP 的 local_rank（用于确定当前进程的GPU）
        
    Returns:
        imputator: Imputator 实例（位于指定设备，如果use_ddp=True则用DDP包裹）
    """
    # -----------------------------------------------------------
    # 1. 确定目标设备（DDP模式下每个进程使用自己的GPU）
    # -----------------------------------------------------------
    try:
        import torch.distributed as dist
        is_ddp = dist.is_initialized()
    except:
        is_ddp = False
    
    if use_ddp or is_ddp:
        # DDP 模式：强制每个进程把 imputator 放在本进程对应的 GPU 上，避免全部挤在 cuda:0
        # （imputator 不参与训练，仅做推理辅助，每个进程一份副本放在本进程 GPU 即可）
        if local_rank is None and is_ddp:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        # 始终按 local_rank 决定设备，忽略调用方传入的 target_device，防止误传 cuda:0 导致全在一卡
        if local_rank is not None:
            target_device = f'cuda:{local_rank}'
        else:
            target_device = target_device or 'cuda:0'
    else:
        # 非DDP模式：使用指定的target_device或第一个可见的GPU（cuda:0）
        if target_device is None:
            target_device = 'cuda:0'
    
    # 只在主进程输出（避免DDP重复输出）
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"🎯 Imputator 将被加载到: {target_device} (DDP模式: {use_ddp or is_ddp})")
    except:
        print(f"🎯 Imputator 将被加载到: {target_device} (DDP模式: {use_ddp or is_ddp})")

    # -----------------------------------------------------------
    # 2. 初始化 Model (在 CPU 上)
    #    如果 configs 含 imp_d_model 等字段（TED 训练时），用它们覆盖主模型参数
    # -----------------------------------------------------------
    import copy
    imp_configs = copy.copy(configs)
    if hasattr(configs, 'imp_d_model'):
        imp_configs.d_model = configs.imp_d_model
        imp_configs.n_heads = getattr(configs, 'imp_n_heads', 8)
        imp_configs.e_layers = getattr(configs, 'imp_e_layers', 6)
        imp_configs.d_ff = getattr(configs, 'imp_d_ff', 1024)
    # Imputator 的 storage 数须与 checkpoint 一致；默认 2（与 --imp_n_storage_tokens 一致）
    _ims = int(getattr(configs, 'imp_n_storage_tokens', 2))
    imp_configs.imp_n_storage_tokens = _ims if _ims >= 0 else 2
    full_model = Model(imp_configs)
    
    # -----------------------------------------------------------
    # 3. 加载权重 (强制 map_location='cpu')
    # -----------------------------------------------------------
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    if os.path.isdir(checkpoint_path):
        inner = os.path.join(checkpoint_path, "checkpoint.pth")
        if not os.path.isfile(inner):
            raise FileNotFoundError(
                f"Expected checkpoint.pth inside directory: {checkpoint_path}"
            )
        checkpoint_path = inner

    # 只在主进程输出（避免DDP重复输出）
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Loading checkpoint from: {checkpoint_path}")
    except:
        print(f"Loading checkpoint from: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # -----------------------------------------------------------
    # 4. 处理 DataParallel/DDP (module.) 前缀
    # -----------------------------------------------------------
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # 去掉 'module.'
        else:
            name = k
        new_state_dict[name] = v
        
    # -----------------------------------------------------------
    # 5. 加载权重到 Model
    # -----------------------------------------------------------
    msg = full_model.load_state_dict(new_state_dict, strict=False)
    # 只在主进程输出（避免DDP重复输出）
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Imputator Load Status: {msg}")
    except:
        print(f"Imputator Load Status: {msg}")
    
    # -----------------------------------------------------------
    # 6. 提取核心组件 & 冻结
    # -----------------------------------------------------------
    imputator = full_model
    imputator.eval()
    for param in imputator.parameters():
        param.requires_grad = False
    
    # -----------------------------------------------------------
    # 7. 移动到目标 GPU
    # -----------------------------------------------------------
    imputator = imputator.to(target_device)
    
    # -----------------------------------------------------------
    # 8. 注意：不使用 DDP 包裹，因为 imputator 的参数被冻结了
    #    但是每个进程已经在自己的 GPU 上加载了 imputator，
    #    这样可以分散显存占用，同时避免 DDP 的错误
    # -----------------------------------------------------------
    # 关键：imputator 的参数被冻结（requires_grad=False），
    # 所以不能使用 DDP 包裹。但每个进程已经在自己的 GPU 上加载了模型，
    # 这已经实现了显存分散的目标。
    if (use_ddp or is_ddp) and local_rank is not None:
        # 只在主进程输出
        try:
            import torch.distributed as dist
            if dist.get_rank() == 0:
                print(f"✅ Imputator 已分散到各 GPU（每个进程在自己的 GPU 上运行，不使用 DDP 包裹）")
        except:
            pass
        
    return imputator