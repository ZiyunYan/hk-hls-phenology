"""Shared AMP autocast helpers for train / val inference."""

from __future__ import annotations

import contextlib
from typing import Optional

import torch
from torch.amp import autocast as torch_amp_autocast


def resolve_amp_dtype(args) -> Optional[torch.dtype]:
    """Effective autocast dtype when use_amp and CUDA; None if AMP disabled."""
    if not (getattr(args, "use_amp", False) and torch.cuda.is_available()):
        return None
    mode = str(getattr(args, "amp_dtype", "auto")).lower()
    if mode == "auto":
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            return torch.float16
    if mode == "bfloat16":
        return torch.bfloat16
    return torch.float16


def amp_autocast_ctx(args):
    """Context manager: BF16/FP16 autocast under --use_amp, else nullcontext."""
    dt = resolve_amp_dtype(args)
    if dt is None:
        return contextlib.nullcontext()
    return torch_amp_autocast("cuda", dtype=dt)
