"""Lightweight condition-routing probe for training (same-z / diff-z CE matrices)."""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from scripts.validate_evidence_gap_condition_relation import (
    compute_same_z_ce_matrix,
    enumerate_teacher_specs_containing_short,
    routing_metrics_from_ce_matrix,
)
from utils.tools import generate_local_view_crop, get_student_input, get_teacher_input


def _teacher_probs(t_logits: torch.Tensor, teacher_temp: float, *, use_sinkhorn: bool) -> torch.Tensor:
    if use_sinkhorn:
        flat = t_logits.reshape(-1, t_logits.shape[-1]).float()
        q = torch.exp(flat / float(teacher_temp)).t()
        k_dim = q.shape[0]
        b_local = q.shape[1]
        b_total = max(float(b_local), 1.0)
        sum_q = torch.sum(q)
        q = q / sum_q
        for _ in range(3):
            sum_of_rows = torch.sum(q, dim=1, keepdim=True)
            q = q / sum_of_rows
            q = q / k_dim
            sum_of_cols = torch.sum(q, dim=0, keepdim=True)
            q = q / sum_of_cols
            q = q / b_total
        q = q * b_total
        return q.t().view_as(t_logits)
    return F.softmax(t_logits.float() / float(teacher_temp), dim=-1)


def _local_dino_ce(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    teacher_temp: float,
    student_temp: float,
    *,
    use_sinkhorn: bool,
) -> float:
    if s_logits.numel() == 0 or t_logits.numel() == 0:
        return float("nan")
    t_prob = _teacher_probs(t_logits, teacher_temp, use_sinkhorn=use_sinkhorn)
    log_p = F.log_softmax(s_logits.float() / float(student_temp), dim=-1)
    return float(-(t_prob.unsqueeze(0) * log_p).sum(dim=-1).mean().item())


def _build_probe_condition(
    model,
    abs_starts: torch.Tensor,
    short_len: int,
    short_tokens: int,
    teacher_start: int,
    teacher_len: int,
    timeline_len: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if getattr(model, "evidence_gap_version", "v2") == "v3":
        student_start = int(abs_starts.reshape(-1)[0].item())
        teacher_tokens = model._num_patches_for_len(int(teacher_len))
        return model._build_evidence_gap_condition_v3(
            int(teacher_start),
            int(teacher_len),
            student_start,
            int(short_len),
            int(short_tokens),
            int(teacher_tokens),
            int(timeline_len),
            device,
            int(batch_size),
        )
    if getattr(model, "evidence_gap_version", "v2") == "v4":
        return model._build_v4_condition_num(
            abs_starts,
            int(short_len),
            int(teacher_start),
            int(teacher_len),
            int(timeline_len),
        )
    return model._build_condition_for_short_in_teacher(
        short_abs_starts=abs_starts,
        short_len=int(short_len),
        short_tokens=int(short_tokens),
        teacher_start=int(teacher_start),
        teacher_len=int(teacher_len),
    )


def _parse_teacher_lengths(model, timeline_len: int) -> list[int]:
    return sorted(
        int(v)
        for v in getattr(model, "evidence_gap_teacher_lengths", [])
        if 0 < int(v) <= int(timeline_len)
    )


@torch.no_grad()
def forward_same_z_diff_t_pack(
    model,
    x_enc: torch.Tensor,
    time_mark: torch.Tensor | None,
    *,
    max_teachers: int = 4,
    max_samples: int = 1,
    generator: torch.Generator | None = None,
) -> dict | None:
    """Fix one short z; multiple (c_i, t_i) from teacher windows containing the same crop."""
    device = x_enc.device
    B, T, _ = x_enc.shape
    teacher_lengths = _parse_teacher_lengths(model, T)
    if not teacher_lengths:
        teacher_lengths = [T]

    choice_lens = teacher_lengths
    teacher_len = int(
        choice_lens[
            int(torch.randint(len(choice_lens), (1,), device="cpu", generator=generator).item())
        ]
    )
    teacher_start = 0
    if teacher_len < T:
        teacher_start = int(
            torch.randint(T - teacher_len + 1, (1,), device="cpu", generator=generator).item()
        )

    short_aug = str(model.evidence_gap_student_aug)
    missing_mask_orig = torch.isnan(x_enc).any(dim=-1)
    x_clean = x_enc.nan_to_num(0.0)

    def _slice(ten, start, length):
        return ten[:, start : start + length] if ten is not None else None

    teacher_tokens = model._num_patches_for_len(teacher_len)
    r_min = float(model.evidence_gap_student_ratio_min)
    r_max = float(model.evidence_gap_student_ratio_max)
    if teacher_tokens <= 1:
        short_tokens = 1
        short_len = teacher_len
    else:
        min_short = max(1, int(math.ceil(r_min * teacher_tokens)))
        max_short = max(min_short, min(teacher_tokens - 1, int(math.floor(r_max * teacher_tokens))))
        short_tokens = int(
            torch.randint(min_short, max_short + 1, (1,), device="cpu", generator=generator).item()
        )
        short_len = min(teacher_len, model.patch_len + (short_tokens - 1) * model.stride)
        short_tokens = int(model._num_patches_for_len(short_len))

    x_teacher = _slice(x_clean, teacher_start, teacher_len)
    missing_teacher = _slice(missing_mask_orig, teacher_start, teacher_len).float()
    time_teacher = _slice(time_mark, teacher_start, teacher_len)

    valid_ratio = (~missing_mask_orig[:, teacher_start : teacher_start + teacher_len]).float().mean(dim=-1)
    valid_idx = torch.where(valid_ratio >= float(model.valid_sample_threshold))[0]
    if valid_idx.numel() == 0:
        return None

    if int(max_samples) > 0 and valid_idx.numel() > int(max_samples):
        pick = valid_idx[: int(max_samples)]
    else:
        pick = valid_idx

    x_teacher = x_teacher.index_select(0, pick)
    missing_teacher = missing_teacher.index_select(0, pick)
    time_teacher = time_teacher.index_select(0, pick) if time_teacher is not None else None
    x_valid = x_clean.index_select(0, pick)
    time_valid = time_mark.index_select(0, pick) if time_mark is not None else None
    miss_valid = missing_mask_orig.index_select(0, pick)
    Bv = int(pick.numel())

    x_crop, crop_starts = generate_local_view_crop(x_teacher, short_len, device)
    batch_idx = torch.arange(Bv, device=device).unsqueeze(1)
    time_idx = crop_starts.unsqueeze(1) + torch.arange(short_len, device=device).unsqueeze(0)
    x_short = get_student_input(x_crop, short_aug, is_train=False)
    miss_short = missing_teacher[batch_idx, time_idx]
    time_short = time_teacher[batch_idx, time_idx] if time_teacher is not None else None

    with autocast(enabled=device.type == "cuda"):
        out_short = model.backbone(
            x_short,
            miss_short.float(),
            time_short,
            mask_map=None,
            is_student=True,
            lon_lat=None,
            geo_keep=None,
        )
    z_anchor = out_short["z_cls"].view(Bv, -1)
    abs_starts = teacher_start + crop_starts.long()
    view_anchor = torch.zeros(Bv, device=device, dtype=torch.long)

    teacher_specs = enumerate_teacher_specs_containing_short(T, abs_starts, short_len, teacher_lengths)
    if int(max_teachers) > 0:
        teacher_specs = teacher_specs[: int(max_teachers)]
    if len(teacher_specs) < 2:
        return None

    t_logits_list = []
    cond_list = []
    teacher_meta = []
    for start, tlen in teacher_specs:
        x_t = _slice(x_valid, start, tlen)
        miss_t = _slice(miss_valid, start, tlen).float()
        time_t = _slice(time_valid, start, tlen)
        x_t_in = get_teacher_input(x_t, "none", is_train=False)
        with autocast(enabled=device.type == "cuda"):
            out_t = model.teacher(
                x_t_in,
                miss_t,
                time_t,
                mask_map=None,
                is_student=False,
                lon_lat=None,
                geo_keep=None,
            )
        t_logits_list.append(out_t["logits_global"].detach())
        cond_list.append(
            _build_probe_condition(
                model,
                abs_starts,
                int(short_len),
                int(short_tokens),
                int(start),
                int(tlen),
                int(T),
                device,
                Bv,
            )
        )
        teacher_meta.append({"teacher_start": int(start), "teacher_len": int(tlen)})

    logits_by_cond = []
    for c_j in cond_list:
        s_j = model._conditioned_evidence_gap_logits(
            z_anchor.unsqueeze(0),
            c_j.unsqueeze(0),
            view_anchor.unsqueeze(0),
        ).squeeze(0)
        logits_by_cond.append(s_j)

    return {
        "Bv": Bv,
        "short_len": int(short_len),
        "short_tokens": int(short_tokens),
        "z_anchor": z_anchor,
        "teacher_specs": teacher_meta,
        "t_logits_list": t_logits_list,
        "cond_list": cond_list,
        "logits_by_cond": logits_by_cond,
        "primary_teacher_len": teacher_len,
        "primary_teacher_start": teacher_start,
    }


def _sample_distinct_crop_starts(
    teacher_len: int,
    short_len: int,
    k: int,
    *,
    min_gap: int,
    generator: torch.Generator | None,
) -> list[int]:
    if short_len >= teacher_len:
        return [0] * k
    max_start = teacher_len - short_len
    starts: list[int] = []
    for _ in range(64):
        if len(starts) >= k:
            break
        s = int(torch.randint(0, max_start + 1, (1,), device="cpu", generator=generator).item())
        if all(abs(s - t) >= min_gap for t in starts):
            starts.append(s)
    return starts


@torch.no_grad()
def forward_diff_z_same_t_pack(
    model,
    x_enc: torch.Tensor,
    time_mark: torch.Tensor | None,
    *,
    max_views: int = 4,
    max_samples: int = 1,
    generator: torch.Generator | None = None,
) -> dict | None:
    """Multiple short crops (different s_i, c_i) on one teacher window; shared teacher target t."""
    device = x_enc.device
    B, T, _ = x_enc.shape
    teacher_lengths = _parse_teacher_lengths(model, T)
    if not teacher_lengths:
        teacher_lengths = [T]

    choice_lens = teacher_lengths
    teacher_len = int(
        choice_lens[
            int(torch.randint(len(choice_lens), (1,), device="cpu", generator=generator).item())
        ]
    )
    teacher_start = 0
    if teacher_len < T:
        teacher_start = int(
            torch.randint(T - teacher_len + 1, (1,), device="cpu", generator=generator).item()
        )

    short_aug = str(model.evidence_gap_student_aug)
    missing_mask_orig = torch.isnan(x_enc).any(dim=-1)
    x_clean = x_enc.nan_to_num(0.0)

    def _slice(ten, start, length):
        return ten[:, start : start + length] if ten is not None else None

    teacher_tokens = model._num_patches_for_len(teacher_len)
    r_min = float(model.evidence_gap_student_ratio_min)
    r_max = float(model.evidence_gap_student_ratio_max)
    if teacher_tokens <= 1:
        short_tokens = 1
        short_len = teacher_len
    else:
        min_short = max(1, int(math.ceil(r_min * teacher_tokens)))
        max_short = max(min_short, min(teacher_tokens - 1, int(math.floor(r_max * teacher_tokens))))
        short_tokens = int(
            torch.randint(min_short, max_short + 1, (1,), device="cpu", generator=generator).item()
        )
        short_len = min(teacher_len, model.patch_len + (short_tokens - 1) * model.stride)
        short_tokens = int(model._num_patches_for_len(short_len))

    x_teacher = _slice(x_clean, teacher_start, teacher_len)
    missing_teacher = _slice(missing_mask_orig, teacher_start, teacher_len).float()
    time_teacher = _slice(time_mark, teacher_start, teacher_len)

    valid_ratio = (~missing_mask_orig[:, teacher_start : teacher_start + teacher_len]).float().mean(dim=-1)
    valid_idx = torch.where(valid_ratio >= float(model.valid_sample_threshold))[0]
    if valid_idx.numel() == 0:
        return None

    if int(max_samples) > 0 and valid_idx.numel() > int(max_samples):
        pick = valid_idx[: int(max_samples)]
    else:
        pick = valid_idx
    Bv = int(pick.numel())

    x_teacher = x_teacher.index_select(0, pick)
    missing_teacher = missing_teacher.index_select(0, pick)
    time_teacher = time_teacher.index_select(0, pick) if time_teacher is not None else None
    x_valid = x_clean.index_select(0, pick)
    time_valid = time_mark.index_select(0, pick) if time_mark is not None else None
    miss_valid = missing_mask_orig.index_select(0, pick)

    k = max(2, int(max_views))
    min_gap = max(1, short_len // 4)
    crop_starts = _sample_distinct_crop_starts(teacher_len, short_len, k, min_gap=min_gap, generator=generator)
    if len(crop_starts) < 2:
        return None
    k = len(crop_starts)

    z_list = []
    cond_list = []
    view_list = []
    student_meta = []
    for rel_start in crop_starts:
        abs_start = teacher_start + int(rel_start)
        rel = int(rel_start)
        time_idx = torch.arange(rel, rel + short_len, device=device).unsqueeze(0).expand(Bv, -1)
        batch_idx = torch.arange(Bv, device=device).unsqueeze(1)
        x_short = x_teacher[:, rel : rel + short_len]
        x_short = get_student_input(x_short, short_aug, is_train=False)
        miss_short = missing_teacher[batch_idx, time_idx]
        time_short = time_teacher[batch_idx, time_idx] if time_teacher is not None else None
        with autocast(enabled=device.type == "cuda"):
            out_short = model.backbone(
                x_short,
                miss_short.float(),
                time_short,
                mask_map=None,
                is_student=True,
                lon_lat=None,
                geo_keep=None,
            )
        z_list.append(out_short["z_cls"].view(Bv, -1))
        abs_starts = torch.full((Bv,), abs_start, device=device, dtype=torch.long)
        cond_list.append(
            _build_probe_condition(
                model,
                abs_starts,
                int(short_len),
                int(short_tokens),
                int(teacher_start),
                int(teacher_len),
                int(T),
                device,
                Bv,
            )
        )
        view_list.append(torch.zeros(Bv, device=device, dtype=torch.long))
        student_meta.append({"student_start": int(abs_start), "student_len": int(short_len)})

    x_t = _slice(x_valid, teacher_start, teacher_len)
    miss_t = _slice(miss_valid, teacher_start, teacher_len).float()
    time_t = _slice(time_valid, teacher_start, teacher_len)
    x_t_in = get_teacher_input(x_t, "none", is_train=False)
    with autocast(enabled=device.type == "cuda"):
        out_t = model.teacher(
            x_t_in,
            miss_t,
            time_t,
            mask_map=None,
            is_student=False,
            lon_lat=None,
            geo_keep=None,
        )
    t_logits = out_t["logits_global"].detach()

    return {
        "Bv": Bv,
        "teacher_start": int(teacher_start),
        "teacher_len": int(teacher_len),
        "short_len": int(short_len),
        "short_tokens": int(short_tokens),
        "z_list": z_list,
        "cond_list": cond_list,
        "view_list": view_list,
        "student_specs": student_meta,
        "t_logits": t_logits,
    }


@torch.no_grad()
def compute_diff_z_same_t_ce_matrix(
    model,
    pack: dict,
    teacher_temp: float,
    student_temp: float,
    *,
    use_sinkhorn: bool,
) -> np.ndarray:
    k = len(pack["z_list"])
    ce_mat = np.full((k, k), np.nan, dtype=np.float64)
    t_logits = pack["t_logits"]
    for i in range(k):
        z_i = pack["z_list"][i]
        for j in range(k):
            c_j = pack["cond_list"][j]
            view_j = pack["view_list"][j]
            s_ij = model._conditioned_evidence_gap_logits(
                z_i.unsqueeze(0),
                c_j.unsqueeze(0),
                view_j.unsqueeze(0),
            ).squeeze(0)
            ce_mat[i, j] = _local_dino_ce(
                s_ij.unsqueeze(0),
                t_logits,
                teacher_temp,
                student_temp,
                use_sinkhorn=use_sinkhorn,
            )
    return ce_mat


def _summarize_ce_matrices(matrices: list[np.ndarray]) -> dict[str, Any]:
    if not matrices:
        return {"n_batches": 0}
    row_accs: list[float] = []
    margins: list[float] = []
    k_vals: list[int] = []
    for m in matrices:
        stats = routing_metrics_from_ce_matrix(m)
        row_accs.append(float(stats["row_argmin_accuracy"]))
        margins.append(float(stats["margin_offdiag_minus_diag"]))
        k_vals.append(int(m.shape[0]))
    k_mean = float(np.mean(k_vals)) if k_vals else float("nan")
    return {
        "n_batches": int(len(matrices)),
        "k_mean": k_mean,
        "random_baseline": float(1.0 / k_mean) if k_mean > 0 else float("nan"),
        "row_argmin_accuracy_mean": float(np.mean(row_accs)),
        "margin_offdiag_minus_diag_mean": float(np.mean(margins)),
    }


def _append_jsonl(path: str, record: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_probe_enabled(args) -> bool:
    flag = int(getattr(args, "probe_condition_routing_enable", -1))
    if flag >= 0:
        return bool(flag)
    return bool(int(getattr(args, "evidence_gap_distill", 0))) and bool(
        int(getattr(args, "evidence_gap_condition", 0))
    )


@torch.no_grad()
def run_lightweight_condition_routing_probe(
    model,
    args,
    device,
    *,
    train_loader,
    epoch: int | None = None,
) -> dict[str, Any] | None:
    if not _resolve_probe_enabled(args):
        return None
    if not getattr(model, "evidence_gap_condition", False):
        return None
    if train_loader is None:
        return None

    num_batches = max(1, int(getattr(args, "probe_condition_routing_batches", 2)))
    max_k = max(2, int(getattr(args, "probe_condition_routing_max_k", 4)))
    max_samples = max(1, int(getattr(args, "probe_condition_routing_max_samples", 1)))
    teacher_temp = float(getattr(args, "teacher_temp", 0.07))
    student_temp = float(getattr(args, "student_temp", 0.1))
    use_sinkhorn = bool(int(getattr(args, "probe_condition_routing_use_sinkhorn", 1)))

    was_training = model.training
    model.eval()
    if hasattr(model, "teacher"):
        model.teacher.eval()
    if hasattr(model, "backbone"):
        model.backbone.eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(getattr(args, "probe_condition_routing_seed", 2026)) + int(epoch or 0))

    same_z_mats: list[np.ndarray] = []
    diff_z_mats: list[np.ndarray] = []
    tried = 0
    try:
        for batch_tuple in train_loader:
            tried += 1
            batch_x = batch_tuple[0].float().to(device, non_blocking=True)
            batch_x_mark = batch_tuple[1]
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.float().to(device, non_blocking=True)

            pack_a = forward_same_z_diff_t_pack(
                model,
                batch_x,
                batch_x_mark,
                max_teachers=max_k,
                max_samples=max_samples,
                generator=generator,
            )
            if pack_a is not None:
                ce_a, _ = compute_same_z_ce_matrix(
                    pack_a,
                    teacher_temp,
                    student_temp,
                    use_sinkhorn=use_sinkhorn,
                    sinkhorn_batch_multiplier=1,
                )
                same_z_mats.append(ce_a)

            pack_b = forward_diff_z_same_t_pack(
                model,
                batch_x,
                batch_x_mark,
                max_views=max_k,
                max_samples=max_samples,
                generator=generator,
            )
            if pack_b is not None:
                ce_b = compute_diff_z_same_t_ce_matrix(
                    model,
                    pack_b,
                    teacher_temp,
                    student_temp,
                    use_sinkhorn=use_sinkhorn,
                )
                diff_z_mats.append(ce_b)

            if len(same_z_mats) >= num_batches and len(diff_z_mats) >= num_batches:
                break
            if tried >= num_batches * 4:
                break
    finally:
        if was_training:
            model.train()

    summary = {
        "epoch": int(epoch + 1) if epoch is not None else None,
        "evidence_gap_version": str(getattr(model, "evidence_gap_version", "v2")),
        "same_z_diff_t": _summarize_ce_matrices(same_z_mats),
        "diff_z_same_t": _summarize_ce_matrices(diff_z_mats),
    }

    if int(getattr(args, "local_rank", 0)) == 0:
        sz = summary["same_z_diff_t"]
        dz = summary["diff_z_same_t"]
        print("\n>>> [Condition Routing Probe]")
        if sz.get("n_batches", 0) > 0:
            print(
                f"    same_z_diff_t (fixed z, diff c_i/t_i): "
                f"argmin_acc={sz['row_argmin_accuracy_mean']:.3f} "
                f"(random~{sz['random_baseline']:.3f}), "
                f"margin={sz['margin_offdiag_minus_diag_mean']:.4f}, "
                f"k~{sz['k_mean']:.1f}, n={sz['n_batches']}"
            )
        else:
            print("    same_z_diff_t: no valid batches (need >=2 teacher windows containing crop)")
        if dz.get("n_batches", 0) > 0:
            print(
                f"    diff_z_same_t (diff s_i/c_i, same t): "
                f"argmin_acc={dz['row_argmin_accuracy_mean']:.3f} "
                f"(random~{dz['random_baseline']:.3f}), "
                f"margin={dz['margin_offdiag_minus_diag_mean']:.4f}, "
                f"k~{dz['k_mean']:.1f}, n={dz['n_batches']}"
            )
        else:
            print("    diff_z_same_t: no valid batches (need >=2 distinct short crops)")

        log_path_cfg = str(getattr(args, "probe_condition_routing_log_file", "auto"))
        if log_path_cfg == "auto":
            logs_dir = os.path.join(".", "logs")
            safe_model_id = str(getattr(args, "model_id", "train")).replace("/", "_")
            log_path = os.path.join(logs_dir, f"{safe_model_id}_condition_routing_probe.jsonl")
        elif os.path.isabs(log_path_cfg):
            log_path = log_path_cfg
        else:
            log_path = os.path.join(".", log_path_cfg)
        _append_jsonl(log_path, summary)

    return summary
