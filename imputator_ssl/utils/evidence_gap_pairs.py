"""
Unified evidence-gap training pairs (draft).

One training pair:
    (student evidence V, target window W, condition c(V,W))
        ->  p_theta(k | z(V), c(V,W))  ~  q_teacher(k | W)

Form A (multi evidence, one target):
    (V1, W0, c1) -> q(W0)
    (V2, W0, c2) -> q(W0)

Form B (one evidence, multi target):
    (V0, W1, c1) -> q(W1)
    (V0, W2, c2) -> q(W2)

Loss in utils/losses.py already supports this via:
    evidence_gap_pairwise=True
    evidence_gap_teacher_row_indices  # pair i -> row of unique teacher logits

Integration plan:
    v4 wired in models/TED_modular._forward_evidence_gap_v4 (A=v2 anchor, B=extra W1).
    Remaining: migrate v2/v3 to shared pair sampler in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch


class ViewType(IntEnum):
    GLOBAL = -1
    CROP = 0
    RANDOM = 1
    DISJOINT = 2  # legacy disjoint_pred view embed id


class PairSource(str):
    FORM_A = "form_a"  # multi V, single W
    FORM_B = "form_b"  # single V (or reused z), multi W


@dataclass(frozen=True)
class WindowSpec:
    """Absolute window on the full timeline [start, start + length)."""

    start: int
    length: int

    def key(self) -> Tuple[int, int]:
        return (int(self.start), int(self.length))

    def end(self) -> int:
        return int(self.start) + int(self.length)


@dataclass
class EvidencePair:
    """
    One distillation pair before tensor batching.

    v: student evidence window (what goes through student backbone, unless z_reuse_row set).
    w: target teacher window (teacher forward -> q_teacher(k|W)).
    view_type: crop/random/global for condition readout embedding.
    source: form_a or form_b (logging / row weights only).
    weight: per-pair loss multiplier (default 1.0).
    z_reuse_row: if set, skip student forward for this pair and reuse z from that row
        (Gate v2 sameShortMT / disjoint_pred anchor path).
    meta: free-form debug tags (relation_id, group_idx, ...).
    """

    v: WindowSpec
    w: WindowSpec
    view_type: int = int(ViewType.CROP)
    source: str = PairSource.FORM_A
    weight: float = 1.0
    z_reuse_row: Optional[int] = None
    meta: Dict[str, Union[int, float, str]] = field(default_factory=dict)

    def needs_student_forward(self) -> bool:
        return self.z_reuse_row is None


@dataclass
class TeacherTargetTable:
    """
    Unique teacher windows W and their row index in t_logits_unique [Tw, B, K].
    """

    windows: List[WindowSpec]
    index: Dict[Tuple[int, int], int]

    @classmethod
    def from_pairs(cls, pairs: Sequence[EvidencePair]) -> "TeacherTargetTable":
        windows: List[WindowSpec] = []
        index: Dict[Tuple[int, int], int] = {}
        for pair in pairs:
            key = pair.w.key()
            if key not in index:
                index[key] = len(windows)
                windows.append(pair.w)
        return cls(windows=windows, index=index)

    def row_for(self, w: WindowSpec) -> int:
        return int(self.index[w.key()])

    def row_indices(self, pairs: Sequence[EvidencePair]) -> List[int]:
        return [self.row_for(p.w) for p in pairs]

    def group_by_length(self) -> Dict[int, List[WindowSpec]]:
        groups: Dict[int, List[WindowSpec]] = {}
        for w in self.windows:
            groups.setdefault(int(w.length), []).append(w)
        return groups


@dataclass
class PackedPairBatch:
    """
    Tensor bundle ready for losses.evidence_gap_pairwise CE.

    Shapes (before valid-sample filtering):
        s_logits: [R, B, K]
        t_logits_unique: [Tw, B, K]
        teacher_row_indices: length R, maps student row -> teacher row
        row_weights: [R, B]
    """

    s_logits: torch.Tensor
    t_logits_unique: torch.Tensor
    teacher_row_indices: torch.Tensor
    row_weights: torch.Tensor
    pairs: List[EvidencePair]
    teacher_table: TeacherTargetTable
    condition_log: Dict[str, Union[int, float, str, torch.Tensor]]


# ---------------------------------------------------------------------------
# Pair list helpers
# ---------------------------------------------------------------------------


def append_form_a_pairs(
    pairs: List[EvidencePair],
    w0: WindowSpec,
    student_windows: Iterable[WindowSpec],
    *,
    view_types: Optional[Sequence[int]] = None,
    weight: float = 1.0,
    meta: Optional[Dict[str, Union[int, float, str]]] = None,
) -> None:
    """Form A: multiple V, single target W0."""
    meta = dict(meta or {})
    for i, v in enumerate(student_windows):
        vt = int(view_types[i]) if view_types is not None else int(ViewType.CROP)
        pairs.append(
            EvidencePair(
                v=v,
                w=w0,
                view_type=vt,
                source=PairSource.FORM_A,
                weight=weight,
                meta={**meta, "form": "A", "v_idx": i},
            )
        )


def append_form_b_pairs(
    pairs: List[EvidencePair],
    v_anchor: WindowSpec,
    target_windows: Iterable[WindowSpec],
    *,
    view_type: int = int(ViewType.CROP),
    z_reuse_row: Optional[int] = None,
    weight: float = 1.0,
    meta: Optional[Dict[str, Union[int, float, str]]] = None,
) -> None:
    """Form B: one evidence V (or reused z), multiple targets Wi."""
    meta = dict(meta or {})
    for i, w in enumerate(target_windows):
        pairs.append(
            EvidencePair(
                v=v_anchor,
                w=w,
                view_type=view_type,
                source=PairSource.FORM_B,
                weight=weight,
                z_reuse_row=z_reuse_row,
                meta={**meta, "form": "B", "w_idx": i},
            )
        )


def group_pairs_by_student_length(
    pairs: Sequence[EvidencePair],
) -> Dict[int, List[EvidencePair]]:
    groups: Dict[int, List[EvidencePair]] = {}
    for p in pairs:
        groups.setdefault(int(p.v.length), []).append(p)
    return groups


def student_forward_rows(pairs: Sequence[EvidencePair]) -> List[int]:
    """Row indices that require a new student backbone forward (not z-reuse)."""
    return [i for i, p in enumerate(pairs) if p.needs_student_forward()]


# ---------------------------------------------------------------------------
# Teacher logits / Sinkhorn (draft)
# ---------------------------------------------------------------------------


def stack_teacher_logits(
    logits_by_window: Dict[Tuple[int, int], torch.Tensor],
    table: TeacherTargetTable,
) -> torch.Tensor:
    """
  Stack teacher global logits into [Tw, B, K] using TeacherTargetTable order.

  logits_by_window: (w_start, w_len) -> [B, K]
    """
    rows = []
    for w in table.windows:
        key = w.key()
        if key not in logits_by_window:
            raise KeyError(f"missing teacher logits for window {key}")
        rows.append(logits_by_window[key])
    return torch.stack(rows, dim=0)


def sinkhorn_teacher_probs_global(
    t_logits_unique: torch.Tensor,
    teacher_temp: float,
    sinkhorn_fn: Callable[[torch.Tensor, float], torch.Tensor],
) -> torch.Tensor:
    """
    Current production behavior: joint SK over all unique teacher rows.

    t_logits_unique: [Tw, B, K] -> t_probs same shape
    """
    flat = t_logits_unique.flatten(0, 1).detach().float()
    out = sinkhorn_fn(flat, teacher_temp)
    return out.unflatten(0, t_logits_unique.shape[:2])


def sinkhorn_teacher_probs_by_length(
    t_logits_unique: torch.Tensor,
    table: TeacherTargetTable,
    teacher_temp: float,
    sinkhorn_fn: Callable[[torch.Tensor, float], torch.Tensor],
) -> torch.Tensor:
    """
    Draft alternative: run SK separately per teacher length group, then stitch back.

    Useful ablation if global SK across different W lengths is undesirable.
    """
    probs = torch.empty_like(t_logits_unique, dtype=torch.float32)
    for length, windows in table.group_by_length().items():
        del length  # grouping key only
        row_ids = [table.index[w.key()] for w in windows]
        block = t_logits_unique.index_select(0, torch.tensor(row_ids, device=t_logits_unique.device))
        block_probs = sinkhorn_teacher_probs_global(block, teacher_temp, sinkhorn_fn)
        for local_i, row_id in enumerate(row_ids):
            probs[row_id] = block_probs[local_i]
    return probs


# ---------------------------------------------------------------------------
# Pack for existing loss path
# ---------------------------------------------------------------------------


def build_row_weights(
    pairs: Sequence[EvidencePair],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    per_sample_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    [R, B] row weights from pair.weight; optional per-sample validity mask [B].
    """
    r = len(pairs)
    w = torch.full((r, batch_size), 1.0, device=device, dtype=dtype)
    for i, p in enumerate(pairs):
        if float(p.weight) != 1.0:
            w[i].fill_(float(p.weight))
    if per_sample_valid is not None:
        w = w * per_sample_valid.to(device=device, dtype=dtype).unsqueeze(0)
    return w


def pack_pairs_for_cls_loss(
    *,
    s_logits: torch.Tensor,
    t_logits_unique: torch.Tensor,
    pairs: Sequence[EvidencePair],
    teacher_table: Optional[TeacherTargetTable] = None,
    row_weights: Optional[torch.Tensor] = None,
    condition_log: Optional[Dict[str, Union[int, float, str, torch.Tensor]]] = None,
    extra_cls_data: Optional[Dict] = None,
) -> Dict:
    """
    Build cls_data dict consumed by utils.losses evidence_gap_pairwise branch.

    Does NOT run Sinkhorn; caller passes t_logits_unique, loss applies SK.
    """
    if teacher_table is None:
        teacher_table = TeacherTargetTable.from_pairs(pairs)
    row_indices = teacher_table.row_indices(pairs)
    if row_weights is None:
        row_weights = torch.ones(
            s_logits.shape[0],
            s_logits.shape[1],
            device=s_logits.device,
            dtype=s_logits.dtype,
        )
    teacher_row_indices_t = torch.tensor(
        row_indices, device=s_logits.device, dtype=torch.long
    )
    if int(teacher_row_indices_t.numel()) != int(s_logits.shape[0]):
        raise ValueError(
            "teacher_row_indices length must match student rows: "
            f"{int(teacher_row_indices_t.numel())} vs {int(s_logits.shape[0])}"
        )

    n_form_b = sum(1 for p in pairs if p.source == PairSource.FORM_B)
    n_form_a = sum(1 for p in pairs if p.source == PairSource.FORM_A)
    n_z_reuse = sum(1 for p in pairs if p.z_reuse_row is not None)

    log = {
        "evidence_gap_pair_count": len(pairs),
        "evidence_gap_pair_form_a": n_form_a,
        "evidence_gap_pair_form_b": n_form_b,
        "evidence_gap_pair_z_reuse": n_z_reuse,
        "evidence_gap_n_unique_teacher_windows": len(teacher_table.windows),
    }
    if condition_log:
        log.update(condition_log)

    cls_data = {
        "cls_loss_mode": "evidence_gap",
        "evidence_gap_pairwise": True,
        "evidence_gap_teacher_row_indices": row_indices,
        "evidence_gap_row_weights": row_weights,
        "s_logits_global_valid": s_logits,
        "t_logits_global_valid": t_logits_unique,
        **log,
    }
    if extra_cls_data:
        cls_data.update(extra_cls_data)
    return cls_data


# ---------------------------------------------------------------------------
# Draft forward orchestration (skeleton)
# ---------------------------------------------------------------------------


class EvidenceGapPairForwardDraft:
    """
    Skeleton orchestrator — methods raise NotImplementedError until wired to TED_modular.

    Intended call flow inside Model._forward_evidence_gap_unified:

        draft = EvidenceGapPairForwardDraft(self)
        pairs = []
        draft.sample_primary_teacher_and_form_a(pairs, ...)
        draft.maybe_sample_form_b(pairs, ...)
        table = TeacherTargetTable.from_pairs(pairs)
        t_logits = draft.teacher_forward_unique(table, ...)
        s_logits = draft.student_forward_pairs(pairs, ...)
        cls_data = pack_pairs_for_cls_loss(s_logits=s_logits, t_logits_unique=t_logits, pairs=pairs)
    """

    def __init__(self, model):
        self.model = model

    def sample_primary_teacher_window(self, timeline_len: int, device) -> WindowSpec:
        raise NotImplementedError("wire to existing teacher_len / teacher_start sampling")

    def sample_form_a_short_pairs(
        self,
        pairs: List[EvidencePair],
        w0: WindowSpec,
        timeline_len: int,
        device,
    ) -> None:
        """
        v2: crop/random inside teacher parent.
        v3: crop/random inside relation-dependent parent (see _compute_v3_short_crop_parent).
        """
        raise NotImplementedError

    def sample_form_b_multi_target_pairs(
        self,
        pairs: List[EvidencePair],
        v_anchor: WindowSpec,
        timeline_len: int,
        device,
        *,
        z_reuse_row: Optional[int] = None,
    ) -> None:
        """
        Gate v2 sameShortMT: alt teachers containing anchor short.
        v3 partial/disjoint: target windows Wi where c(V,Wi) is meaningful.
        """
        raise NotImplementedError

    def teacher_forward_unique(
        self,
        table: TeacherTargetTable,
        x_full,
        time_mark_full,
        missing_mask_full,
        lon_lat_full,
        **teacher_kwargs,
    ) -> torch.Tensor:
        """Returns t_logits_unique [Tw, B, K]."""
        raise NotImplementedError

    def student_forward_pairs(
        self,
        pairs: Sequence[EvidencePair],
        x_full,
        time_mark_full,
        missing_mask_full,
        lon_lat_full,
        condition_builder: Callable[[EvidencePair], Tuple[torch.Tensor, int]],
        **student_kwargs,
    ) -> torch.Tensor:
        """
        Returns s_logits [R, B, K].

        condition_builder(pair) -> (condition_num [B, C], view_type scalar or [B])
        For z_reuse_row, apply conditioned head on cached z instead of backbone.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Minimal usage example (documentation)
# ---------------------------------------------------------------------------

def _example_pair_list() -> List[EvidencePair]:
    w0 = WindowSpec(start=100, length=244)
    pairs: List[EvidencePair] = []
    append_form_a_pairs(
        pairs,
        w0,
        [
            WindowSpec(120, 61),
            WindowSpec(140, 61),
            WindowSpec(160, 61),
        ],
        view_types=[ViewType.CROP, ViewType.CROP, ViewType.RANDOM],
    )
    append_form_b_pairs(
        pairs,
        v_anchor=WindowSpec(130, 61),
        target_windows=[
            WindowSpec(0, 244),
            WindowSpec(200, 183),
        ],
        z_reuse_row=1,
    )
    return pairs
