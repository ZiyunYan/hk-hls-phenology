"""Downstream probe helpers for multi-CLS (evidence-gap) models."""
from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F


def num_patches_for_len(length: int, patch_len: int, stride: int) -> int:
    length = int(length)
    if length <= int(patch_len):
        return 1
    return int(math.ceil((length - int(patch_len)) / int(stride)) + 1)


def gap_cls_id_from_gap_tokens(gap_tokens: int, cls_bins) -> int:
    gap_tokens = int(gap_tokens)
    bins = [int(v) for v in cls_bins]
    for idx, upper in enumerate(bins):
        if gap_tokens <= int(upper):
            return idx
    return len(bins)


def parse_int_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    vals = [int(v.strip()) for v in str(value).split(",") if v.strip()]
    return vals if len(vals) > 0 else list(default)


def probe_gap_cls_id(seq_len: int, args: Any) -> int:
    """
    Pick CLS index for downstream encode, mirroring training gap binning.

    Reference teacher length is the full downstream seq_len (default 732).
    Shorter probe windows get a positive token gap → larger cls id.
    """
    patch_len = int(getattr(args, "patch_len", 3))
    stride = int(getattr(args, "stride", 3))
    cls_bins = parse_int_list(
        getattr(args, "evidence_gap_cls_bins", "5,10,21,41,82"),
        default=[5, 10, 21, 41, 82],
    )
    ref_len = int(getattr(args, "seq_len", 732))
    teacher_tokens = num_patches_for_len(ref_len, patch_len, stride)
    student_tokens = num_patches_for_len(int(seq_len), patch_len, stride)
    gap_tokens = max(0, teacher_tokens - student_tokens)
    return gap_cls_id_from_gap_tokens(gap_tokens, cls_bins)


def _resolve_cls_bank(outputs: dict) -> Optional[torch.Tensor]:
    fused = outputs.get("cls_token")
    bank = outputs.get("cls_token_bank")
    if bank is None and fused is not None and fused.ndim == 3:
        bank = fused
    return bank


def _fusion_tau(args: Any) -> float:
    tau = float(getattr(args, "probe_cls_fusion_tau", 1.0))
    return max(tau, 1e-6)


def gap_soft_weights(
    seq_len: int,
    args: Any,
    n_cls: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Gaussian soft weights over CLS indices, centered at estimated gap bin."""
    center = float(probe_gap_cls_id(seq_len, args))
    tau = _fusion_tau(args)
    idx = torch.arange(n_cls, device=device, dtype=dtype)
    logits = -((idx - center) ** 2) / (2.0 * tau * tau)
    return F.softmax(logits, dim=0)


def fuse_cls_gap_soft(
    bank: torch.Tensor,
    seq_len: int,
    args: Any,
) -> torch.Tensor:
    """Weighted sum over CLS bank; full 732-step → peak at CLS#0, tails on others."""
    n_cls = int(bank.shape[1])
    w = gap_soft_weights(seq_len, args, n_cls, bank.device, bank.dtype)
    return (bank * w.view(1, n_cls, 1)).sum(dim=1)


def fuse_cls_attn_readout(
    bank: torch.Tensor,
    patch_tokens: Optional[torch.Tensor],
    args: Any,
) -> torch.Tensor:
    """
    Patch-mean queries CLS bank (content-aware fusion, no extra params).
    Falls back to mean pooling if patch tokens are unavailable.
    """
    if patch_tokens is None:
        return bank.mean(dim=1)
    if patch_tokens.ndim == 4:
        patch_tokens = patch_tokens.mean(dim=1)
    query = patch_tokens.mean(dim=1)
    query = F.normalize(query.float(), dim=-1, eps=1e-6).to(dtype=bank.dtype)
    bank_n = F.normalize(bank.float(), dim=-1, eps=1e-6).to(dtype=bank.dtype)
    scores = (bank_n * query.unsqueeze(1)).sum(dim=-1)
    tau = _fusion_tau(args)
    weights = F.softmax(scores / tau, dim=-1)
    return (bank * weights.unsqueeze(-1)).sum(dim=1)


def fuse_cls_gap0_plus_mean(bank: torch.Tensor) -> torch.Tensor:
    """Concat full-evidence CLS#0 with mean of all heads → [B, 2D]."""
    primary = bank[:, 0]
    pooled = bank.mean(dim=1)
    return torch.cat([primary, pooled], dim=-1)


def resolve_probe_cls_bank(outputs: dict, args: Any) -> torch.Tensor:
    """Return CLS bank [B, K, D] for per-head KNN ensemble."""
    bank = _resolve_cls_bank(outputs)
    if bank is None:
        fused = outputs.get("cls_token")
        if fused is None:
            raise KeyError("encode outputs missing cls_token / cls_token_bank")
        if fused.ndim == 2:
            return fused.unsqueeze(1)
        if fused.ndim == 3:
            return fused
        raise ValueError(f"Unexpected cls_token shape {tuple(fused.shape)}")
    return bank


def resolve_probe_cls_embedding(
    outputs: dict,
    args: Any,
    seq_len: int | None = None,
    mode_override: str | None = None,
) -> torch.Tensor:
    """
    Map model.encode outputs to a single [B, D] or [B, k*D] probe vector.

    probe_cls_mode:
      - gap0: CLS #0 only (full-evidence head)
      - auto_gap: hard pick CLS by token-gap from window length
      - gap_soft: soft Gaussian mix over all CLS, centered at estimated gap bin
      - attn_readout: patch-mean attention over CLS bank
      - gap0_plus_mean: concat [CLS#0, mean(bank)] → 2D dims
      - mean / concat / fused: pooling baselines
      - knn_ensemble: not a vector; use resolve_probe_cls_bank + ensemble KNN
    """
    mode = str(mode_override or getattr(args, "probe_cls_mode", "gap0")).strip().lower()
    if mode == "knn_ensemble":
        raise ValueError(
            "probe_cls_mode=knn_ensemble expects per-CLS bank features; "
            "use resolve_probe_cls_bank() and ExpProbe._eval_knn_cls_bank_ensemble()."
        )

    fused = outputs.get("cls_token")
    bank = _resolve_cls_bank(outputs)

    if bank is None or int(bank.shape[1]) <= 1:
        emb = fused
        if emb is None:
            raise KeyError("encode outputs missing cls_token / cls_token_bank")
        if emb.ndim == 3:
            emb = emb.mean(dim=1)
        return emb

    n_cls = int(bank.shape[1])
    if seq_len is None:
        seq_len = int(getattr(args, "seq_len", 732))

    if mode == "fused":
        if fused is None:
            fused = bank.mean(dim=1)
        return fused
    if mode == "mean":
        return bank.mean(dim=1)
    if mode == "concat":
        return bank.flatten(1)
    if mode == "gap_soft":
        return fuse_cls_gap_soft(bank, seq_len, args)
    if mode == "attn_readout":
        return fuse_cls_attn_readout(bank, outputs.get("patch_tokens"), args)
    if mode in {"gap0_plus_mean", "primary_plus_mean"}:
        return fuse_cls_gap0_plus_mean(bank)
    if mode in {"auto_gap", "auto"}:
        cls_id = probe_gap_cls_id(seq_len, args)
    else:
        cls_id = 0
    cls_id = max(0, min(int(cls_id), n_cls - 1))
    return bank[:, cls_id]
