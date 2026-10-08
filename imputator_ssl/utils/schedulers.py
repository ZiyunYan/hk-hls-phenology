"""
DINOv3 风格的动态参数调度器
支持学习率、weight_decay、teacher_temp、momentum 的动态调整

与 facebookresearch/dinov3 对齐：
- scheduler_version=cosine：CosineScheduler（对应仅 optim.* 的旧配置）
- scheduler_version=dinov3_v2：linear_warmup_cosine_decay 预计算整条曲线（对应 cfg.schedules v2）
"""
from __future__ import annotations

import numpy as np
import math
import torch.distributed as dist


def linear_warmup_cosine_decay(
    start: float,
    peak: float,
    end: float,
    warmup_iterations: int,
    total_iterations: int,
    cosine_iterations: int | None = None,
) -> np.ndarray:
    """
    DINOv3 train.cosine_lr_scheduler.linear_warmup_cosine_decay（与官方一致）。
    顺序：线性 warmup（endpoint=False，长度为 warmup_iterations）→ 余弦段 → 可选常数 end 尾段。
    """
    linear = np.linspace(start, peak, warmup_iterations, endpoint=False)
    if cosine_iterations is None:
        cosine_iterations = total_iterations - warmup_iterations
    cosine_iterations = int(cosine_iterations)
    if cosine_iterations < 1:
        raise ValueError(
            f"linear_warmup_cosine_decay: cosine_iterations={cosine_iterations} invalid "
            f"(warmup_iterations={warmup_iterations}, total_iterations={total_iterations})"
        )
    cosine = np.cos(np.linspace(0, np.pi, cosine_iterations))
    cosine = (cosine + 1) / 2
    cosine = (peak - end) * cosine + end
    remaining_iterations = total_iterations - cosine_iterations - warmup_iterations
    if remaining_iterations < 0:
        raise ValueError(
            f"linear_warmup_cosine_decay: total_iterations={total_iterations} < "
            f"warmup_iterations({warmup_iterations}) + cosine_iterations({cosine_iterations})"
        )
    constant = np.full((remaining_iterations,), fill_value=end, dtype=np.float64)
    return np.concatenate([linear, cosine, constant]).astype(np.float64, copy=False)


class ArraySchedule(object):
    """由 numpy 数组给出的逐 iter 调度（与 CosineScheduler 一样支持 [it] 索引）。"""

    def __init__(self, schedule: np.ndarray):
        self.schedule = np.asarray(schedule, dtype=np.float64)
        self.final_value = float(self.schedule[-1]) if self.schedule.size else 0.0

    def __getitem__(self, it):
        if it >= len(self.schedule):
            return self.final_value
        return float(self.schedule[it])


class CosineScheduler(object):
    """
    Cosine 调度器，支持 warmup 和 cosine decay
    参考 DINOv3 官方实现
    """
    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
        freeze_iters=0,
        trunc_extra=0.0,
    ):
        super().__init__()
        self.final_value = np.float64(final_value)
        self.total_iters = total_iters

        # Freeze 阶段（通常用于最后一层）
        # 安全钳位：freeze + warmup 不能超过总步数
        if freeze_iters + warmup_iters > total_iters:
            warmup_iters = max(0, total_iters - freeze_iters)
            if freeze_iters > total_iters:
                freeze_iters = total_iters
                warmup_iters = 0

        freeze_schedule = np.zeros((freeze_iters))

        # Warmup 阶段（线性增长）
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        # Cosine decay 阶段
        if trunc_extra == 0.0:
            iters = np.arange(total_iters - warmup_iters - freeze_iters)
            schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        else:
            cosine_steps = total_iters - warmup_iters - freeze_iters
            iters = np.linspace(0, np.pi, int((1 + trunc_extra) * cosine_steps))[:cosine_steps]
            schedule = np.cos(iters)
            schedule = (schedule + 1) / 2
            schedule = (schedule - schedule[-1]) / (1 - schedule[-1])
            schedule = schedule * (base_value - final_value) + final_value

        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule), dtype=np.float64)

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        """获取第 it 次迭代的值"""
        if it >= self.total_iters:
            return self.final_value
        else:
            return float(self.schedule[it])


def _get_or(args, key, fallback):
    """getattr that also falls back when the attribute is None."""
    val = getattr(args, key, None)
    return val if val is not None else fallback


def _ddp_world_size(args):
    """World size for LR scaling; matches DINOv3 distributed.get_world_size()."""
    if not getattr(args, "use_multi_gpu", False):
        return 1
    try:
        if dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    if hasattr(args, "devices") and isinstance(args.devices, str):
        n = len([x for x in args.devices.split(",") if x.strip()])
        return max(1, n)
    return 1


def _rank0_should_print(args):
    if not getattr(args, "use_multi_gpu", False):
        return True
    try:
        if dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def _compute_scaled_lr_min_lr(args):
    """按 scaling_rule 缩放 peak / floor LR（与 DINOv3 train.build_schedulers_v2 一致）。"""
    base_lr = args.learning_rate
    scaling_rule = getattr(args, "scaling_rule", "sqrt_wrt_1024")

    if scaling_rule == "sqrt_wrt_1024":
        world_size = _ddp_world_size(args)
        batch_size_per_gpu = args.batch_size
        global_batch_size = batch_size_per_gpu * world_size
        scale_factor = 4 * math.sqrt(global_batch_size / 1024.0)
        scaled_lr = base_lr * scale_factor
        scaled_min_lr = _get_or(args, "min_lr", base_lr * 1e-6) * scale_factor
        if _rank0_should_print(args):
            print(
                f"Learning rate scaling ({scaling_rule}): {base_lr} -> {scaled_lr:.6f} "
                f"(per_gpu={batch_size_per_gpu}, world_size={world_size}, "
                f"global_batch={global_batch_size}, scale={scale_factor:.4f})"
            )
    elif scaling_rule == "linear_wrt_256":
        world_size = _ddp_world_size(args)
        batch_size_per_gpu = args.batch_size
        global_batch_size = batch_size_per_gpu * world_size
        scale_factor = global_batch_size / 256.0
        scaled_lr = base_lr * scale_factor
        scaled_min_lr = _get_or(args, "min_lr", base_lr * 1e-6) * scale_factor
        if _rank0_should_print(args):
            print(
                f"Learning rate scaling ({scaling_rule}): {base_lr} -> {scaled_lr:.6f} "
                f"(per_gpu={batch_size_per_gpu}, world_size={world_size}, "
                f"global_batch={global_batch_size}, scale={scale_factor:.4f})"
            )
    else:
        scaled_lr = base_lr
        scaled_min_lr = _get_or(args, "min_lr", base_lr * 1e-6)

    return scaled_lr, scaled_min_lr


def build_schedulers_dinov3_v2(args, train_steps_per_epoch):
    """
    DINOv3 train.build_schedulers_v2：LR/WD/momentum/teacher_temp 均用 linear_warmup_cosine_decay。
    schedule_trunc_extra 在此分支中不参与（与官方 v2 一致）。
    """
    if getattr(args, "schedule_trunc_extra", 0.0) and _rank0_should_print(args):
        print(
            "[schedulers] scheduler_version=dinov3_v2: ignoring schedule_trunc_extra="
            f"{getattr(args, 'schedule_trunc_extra', 0.0)}"
        )

    total_iterations = int(args.train_epochs * train_steps_per_epoch)
    iter_pe = int(train_steps_per_epoch)
    scaled_lr, scaled_min_lr = _compute_scaled_lr_min_lr(args)

    lr_start = float(_get_or(args, "sched_lr_start", 0.0))
    lr_peak = getattr(args, "sched_lr_peak", None)
    if lr_peak is None:
        lr_peak = scaled_lr
    else:
        lr_peak = float(lr_peak)
    lr_end = getattr(args, "sched_lr_end", None)
    if lr_end is None:
        lr_end = scaled_min_lr
    else:
        lr_end = float(lr_end)
    lr_warmup_epochs = getattr(args, "sched_lr_warmup_epochs", None)
    if lr_warmup_epochs is None:
        lr_warmup_epochs = int(args.warmup_epochs)
    else:
        lr_warmup_epochs = int(lr_warmup_epochs)
    lr_cos_epochs = getattr(args, "sched_lr_cosine_epochs", None)
    lr_cos_iters = int(iter_pe * lr_cos_epochs) if lr_cos_epochs is not None else None

    lr_arr = linear_warmup_cosine_decay(
        start=lr_start,
        peak=lr_peak,
        end=lr_end,
        warmup_iterations=int(iter_pe * lr_warmup_epochs),
        total_iterations=total_iterations,
        cosine_iterations=lr_cos_iters,
    )
    last_layer_lr = lr_arr.copy()
    freeze_e = int(getattr(args, "freeze_last_layer_epochs", 1))
    last_layer_lr[: int(iter_pe * freeze_e)] = 0.0

    wd_s = getattr(args, "sched_wd_start", None)
    wd_p = getattr(args, "sched_wd_peak", None)
    wd_e = getattr(args, "sched_wd_end", None)
    if wd_s is None:
        wd_s = float(args.weight_decay)
    else:
        wd_s = float(wd_s)
    if wd_p is None:
        wd_p = float(args.weight_decay)
    else:
        wd_p = float(wd_p)
    if wd_e is None:
        wd_e = float(_get_or(args, "weight_decay_end", args.weight_decay * 10))
    else:
        wd_e = float(wd_e)
    wd_warm_e = int(getattr(args, "sched_wd_warmup_epochs", 0))
    wd_cos_e = getattr(args, "sched_wd_cosine_epochs", None)
    wd_cos_iters = int(iter_pe * wd_cos_e) if wd_cos_e is not None else None
    wd_arr = linear_warmup_cosine_decay(
        start=wd_s,
        peak=wd_p,
        end=wd_e,
        warmup_iterations=int(iter_pe * wd_warm_e),
        total_iterations=total_iterations,
        cosine_iterations=wd_cos_iters,
    )

    mom_s = getattr(args, "sched_momentum_start", None)
    mom_p = getattr(args, "sched_momentum_peak", None)
    mom_e = getattr(args, "sched_momentum_end", None)
    mom_def_s = float(getattr(args, "momentum_teacher", 0.992))
    mom_def_e = float(getattr(args, "final_momentum_teacher", 1.0))
    if mom_s is None:
        mom_s = mom_def_s
    else:
        mom_s = float(mom_s)
    if mom_p is None:
        mom_p = mom_def_s
    else:
        mom_p = float(mom_p)
    if mom_e is None:
        mom_e = mom_def_e
    else:
        mom_e = float(mom_e)
    mom_warm_e = int(getattr(args, "sched_momentum_warmup_epochs", 0))
    mom_cos_e = getattr(args, "sched_momentum_cosine_epochs", None)
    mom_cos_iters = int(iter_pe * mom_cos_e) if mom_cos_e is not None else None
    mom_arr = linear_warmup_cosine_decay(
        start=mom_s,
        peak=mom_p,
        end=mom_e,
        warmup_iterations=int(iter_pe * mom_warm_e),
        total_iterations=total_iterations,
        cosine_iterations=mom_cos_iters,
    )

    tt_s = float(getattr(args, "warmup_teacher_temp", 0.04))
    tt_peak = float(getattr(args, "teacher_temp", 0.07))
    tt_end = getattr(args, "sched_teacher_temp_end", None)
    if tt_end is None:
        tt_end = tt_peak
    else:
        tt_end = float(tt_end)
    tt_warm_e = int(getattr(args, "warmup_teacher_temp_epochs", 30))
    tt_cos_e = getattr(args, "sched_teacher_temp_cosine_epochs", None)
    tt_cos_iters = int(iter_pe * tt_cos_e) if tt_cos_e is not None else None
    tt_arr = linear_warmup_cosine_decay(
        start=tt_s,
        peak=tt_peak,
        end=tt_end,
        warmup_iterations=int(iter_pe * tt_warm_e),
        total_iterations=total_iterations,
        cosine_iterations=tt_cos_iters,
    )

    return (
        ArraySchedule(lr_arr),
        ArraySchedule(wd_arr),
        ArraySchedule(mom_arr),
        ArraySchedule(tt_arr),
        ArraySchedule(last_layer_lr),
    )


def build_schedulers(args, train_steps_per_epoch):
    """
    构建所有调度器
    
    Args:
        args: 配置参数
        train_steps_per_epoch: 每个 epoch 的迭代次数
    
    Returns:
        lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule
    """
    version = str(getattr(args, "scheduler_version", "cosine")).strip().lower()
    if version in ("dinov3_v2", "v2", "schedules_v2"):
        return build_schedulers_dinov3_v2(args, train_steps_per_epoch)

    total_iters = args.train_epochs * train_steps_per_epoch
    scaled_lr, scaled_min_lr = _compute_scaled_lr_min_lr(args)

    # 1. Learning Rate Schedule
    lr_schedule = CosineScheduler(
        base_value=scaled_lr,
        final_value=scaled_min_lr,
        total_iters=total_iters,
        warmup_iters=args.warmup_epochs * train_steps_per_epoch,
        start_warmup_value=0,
        trunc_extra=getattr(args, 'schedule_trunc_extra', 0.0),
    )
    
    # Last layer LR (前几个 epoch freeze)
    last_layer_lr_schedule = CosineScheduler(
        base_value=scaled_lr,
        final_value=scaled_min_lr,
        total_iters=total_iters,
        warmup_iters=args.warmup_epochs * train_steps_per_epoch,
        start_warmup_value=0,
        freeze_iters=getattr(args, 'freeze_last_layer_epochs', 1) * train_steps_per_epoch,
        trunc_extra=getattr(args, 'schedule_trunc_extra', 0.0),
    )
    
    # 2. Weight Decay Schedule
    weight_decay_start = args.weight_decay
    weight_decay_end = _get_or(args, 'weight_decay_end', args.weight_decay * 10)  # 默认增长 10 倍
    wd_schedule = CosineScheduler(
        base_value=weight_decay_start,
        final_value=weight_decay_end,
        total_iters=total_iters,
        warmup_iters=0,  # weight_decay 通常不需要 warmup
        trunc_extra=getattr(args, 'schedule_trunc_extra', 0.0),
    )
    
    # 3. Momentum Schedule (EMA)
    momentum_start = getattr(args, 'momentum_teacher', 0.992)
    momentum_end = getattr(args, 'final_momentum_teacher', 1.0)
    momentum_schedule = CosineScheduler(
        base_value=momentum_start,
        final_value=momentum_end,
        total_iters=total_iters,
        warmup_iters=0,
        trunc_extra=getattr(args, 'schedule_trunc_extra', 0.0),
    )
    
    # 4. Teacher Temperature Schedule
    warmup_teacher_temp = getattr(args, 'warmup_teacher_temp', 0.04)
    teacher_temp = getattr(args, 'teacher_temp', 0.07)
    warmup_teacher_temp_epochs = getattr(args, 'warmup_teacher_temp_epochs', 30)
    
    teacher_temp_schedule = CosineScheduler(
        base_value=teacher_temp,
        final_value=teacher_temp,  # 最终保持 teacher_temp
        total_iters=warmup_teacher_temp_epochs * train_steps_per_epoch,
        warmup_iters=warmup_teacher_temp_epochs * train_steps_per_epoch,
        start_warmup_value=warmup_teacher_temp,
    )
    
    return lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr, is_last_layer_key="is_last_layer"):
    """
    应用学习率和 weight_decay 到 optimizer

    若某 param_group 设置了 ``scheduled_weight_decay=False``（如 bias/norm/embedding），
    则该组不随 ``wd`` 调度改写，保持创建时的 weight_decay（通常为 0）。

    Args:
        optimizer: 优化器
        lr: 当前学习率
        wd: 当前 weight_decay（仅作用于 ``scheduled_weight_decay`` 非 False 的分组）
        last_layer_lr: 最后一层的学习率（可能被 freeze）
        is_last_layer_key: param_group 中标识最后一层的 key
    """
    for param_group in optimizer.param_groups:
        # 只对参与 WD 调度的分组更新 weight_decay；norm/bias/embedding 等分组保持创建时的 0
        if param_group.get("scheduled_weight_decay", True):
            param_group["weight_decay"] = wd

        # 更新 learning rate
        if is_last_layer_key in param_group and param_group[is_last_layer_key]:
            param_group["lr"] = last_layer_lr
        else:
            param_group["lr"] = lr

