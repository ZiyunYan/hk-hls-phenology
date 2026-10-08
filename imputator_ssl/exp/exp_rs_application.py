import math
import warnings
import os
import io
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# 添加项目根目录到 Python 路径，以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 项目根目录
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, 
                             roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from netCDF4 import Dataset
from utils.timefeatures import time_features
from analysis.analyze_synthetic_trajectories import buildTimeMarkFromTime, applyPreScaling, getDefaultPreScaler
from utils.probe_cls import resolve_probe_cls_embedding, resolve_probe_cls_bank

# 导入项目模块
from data_provider.data_factory import (
    data_provider_CropHarvest_Classification,
    data_provider_LCMAP_Classification,
    data_provider_LCMAP_Segmentation,
    data_provider_GlanceTraining_Classification,
    data_provider_GlobalTree_Classification,
    data_provider_GlobalTree_Segmentation,
    data_provider_GlanceTraining_Segmentation,
    data_provider_CDL_Classification,
)

def parse_alphaearth_years(years_str: str) -> Tuple[int, int]:
    """
    解析 AlphaEarth 年份范围字符串。
    
    Args:
        years_str: 形如 '2017-2021' 的字符串。
    
    Returns:
        (start_year, end_year)
    """
    try:
        parts = years_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid alphaearth_years format: {years_str}")
        start_year = int(parts[0])
        end_year = int(parts[1])
        if start_year > end_year:
            raise ValueError(f"Start year {start_year} > end year {end_year}")
        return start_year, end_year
    except Exception as e:
        print(f">>> [Warning] Failed to parse alphaearth_years '{years_str}', fallback to 2017-2021. Error: {e}")
        return 2017, 2021

# 设置 OpenBLAS 线程数以避免与 sklearn 多线程冲突
# 限制 OpenBLAS 使用单线程，让 sklearn 自己管理线程
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

warnings.filterwarnings('ignore')

# 类级别的缓存，用于存储baseline特征和评估结果
# 键为任务类型和数据集标识，值为缓存的特征和split索引
_baseline_cache = {}


def _filter_small_classes(
    y_all: np.ndarray,
    min_samples_per_class: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    排除「类别总样本数 < min_samples_per_class」的类别，不参与训练与精度计算。
    返回 (idx_keep, y_remap)：idx_keep 为保留样本的索引，y_remap 为重映射后的标签 0..C'-1。
    """
    uniq, counts = np.unique(y_all, return_counts=True)
    valid_classes = uniq[counts >= min_samples_per_class]
    if len(valid_classes) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=y_all.dtype)
    mask = np.isin(y_all, valid_classes)
    idx_keep = np.where(mask)[0]
    y_keep = y_all[mask]
    remap = {c: i for i, c in enumerate(valid_classes)}
    y_remap = np.array([remap[c] for c in y_keep], dtype=np.int64)
    return idx_keep, y_remap


def _alphaearth_data_after_class_filter(
    ae_data: np.ndarray,
    idx_keep: np.ndarray,
    context: str,
) -> np.ndarray:
    """
    AlphaEarth npz 行顺序应与 DataLoader 全量样本一致。小类过滤后 TED 特征用 idx_keep 子集，
    此处必须 ae_data[idx_keep]，不能用 ae_data[:len(y_filtered)]（剔除样本不在末尾时会整表错位）。
    """
    if idx_keep.size == 0:
        return ae_data[:0]
    mx = int(idx_keep.max())
    if mx >= ae_data.shape[0]:
        raise ValueError(
            f"{context}: AlphaEarth npz has N={ae_data.shape[0]} but max(idx_keep)={mx}. "
            "Need one row per pre-filter sample in loader order."
        )
    return ae_data[idx_keep]


def _resolve_first_existing_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _validate_alphaearth_array(ae_data: np.ndarray, context: str) -> np.ndarray:
    """
    基础质量检查：若有效值几乎为 0（全 NaN / Inf），直接报错，避免输出误导性结果。
    """
    finite_mask = np.isfinite(ae_data)
    finite_ratio = float(finite_mask.mean()) if ae_data.size > 0 else 0.0
    if finite_ratio <= 0.0:
        raise ValueError(
            f"{context}: AlphaEarth data has no finite values (all NaN/Inf). "
            "Please replace with a valid GSE file."
        )
    ae_data = np.nan_to_num(ae_data, nan=0.0, posinf=0.0, neginf=0.0)
    return ae_data


def _get_few_shot_split(
    n: int,
    y_all: np.ndarray,
    few_shot_k: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """每类取 few_shot_k 个样本作 train，其余 test。返回 idx_train, idx_test（对应当前 y_all 的索引 0..n-1）。"""
    rng = np.random.RandomState(seed)
    idx_train_list = []
    idx_test_list = []
    indices = np.arange(n)
    for c in np.unique(y_all):
        mask = y_all == c
        idx_c = indices[mask]
        rng.shuffle(idx_c)
        n_c = len(idx_c)
        k = min(few_shot_k, n_c)
        idx_train_list.extend(idx_c[:k])
        idx_test_list.extend(idx_c[k:])
    return np.array(idx_train_list), np.array(idx_test_list)


def _sample_per_class_np(X: np.ndarray, y: np.ndarray, max_per_class: int = 100, seed: int = 2026) -> Tuple[np.ndarray, np.ndarray]:
    """
    每个类别随机采样不超过 max_per_class 个样本，用于 t-SNE 等可视化。
    """
    rng = np.random.RandomState(seed)
    indices = []
    classes = np.unique(y)
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        if cls_idx.size == 0:
            continue
        take = min(max_per_class, cls_idx.size)
        sel = rng.choice(cls_idx, size=take, replace=False)
        indices.append(sel)
    if not indices:
        return X[:0], y[:0]
    idx = np.concatenate(indices)
    return X[idx], y[idx]

# ==========================================
# 纯监督学习模型定义
# ==========================================

class ExpProbe:
    """
    Lightweight External Probe (ExpProbe)
    Function: Forward Inference -> Feature Extraction -> KNN Classification Evaluation
    Optimized for memory efficiency and code reusability.
    
    优化：baseline（原始数据）特征和评估结果会被缓存，避免每个epoch重复计算。
    """

    def __init__(self, device: torch.device):
        self.device = device

    @staticmethod
    def _apply_probe_cloud_mask(
        batch_x: torch.Tensor,
        cloud_mask_ratio: float,
        seed: int,
        step: int,
    ) -> torch.Tensor:
        """
        在 encode 之前对输入序列做“多云”模拟：按时间步随机丢弃一定比例的有效观测点。

        规则：
        - 以「时间步」为单位掩码：一旦选中某个时间步，则该时间步所有通道都置为 NaN；
        - 仅在原本有效（该时间步所有通道都非 NaN）的时间步上采样掩码；
        - 用 seed + step 产生可复现的随机掩码。

        Args:
            batch_x: [B, T, C]
            cloud_mask_ratio: (0, 1) 之间，0 表示不掩码
            seed: 基础随机种子
            step: 当前 batch 的步数（用于让每个 batch 的掩码不同，但整体可复现）
        """
        try:
            ratio = float(cloud_mask_ratio)
        except Exception:
            ratio = 0.0
        if ratio <= 0.0:
            return batch_x
        ratio = min(max(ratio, 0.0), 1.0)

        # valid time step: all channels are not NaN
        valid_ts = ~torch.isnan(batch_x).any(dim=-1)  # [B, T]
        if valid_ts.sum().item() == 0:
            return batch_x

        gen = torch.Generator(device=batch_x.device)
        gen.manual_seed(int(seed) + int(step))
        rand = torch.rand(valid_ts.shape, device=batch_x.device, generator=gen)
        cloud_mask = (rand < ratio) & valid_ts  # [B, T]

        if cloud_mask.any().item():
            out = batch_x.clone()
            out[cloud_mask] = torch.nan
            return out
        return batch_x

    def _probe_cls_bank(
        self,
        outputs: Dict[str, Any],
        args: Any,
    ) -> torch.Tensor:
        return resolve_probe_cls_bank(outputs, args)

    def _append_probe_cls_batch(
        self,
        outputs: Dict[str, Any],
        args: Any,
        seq_len: int,
        embed_list: List[np.ndarray],
        bank_list: Optional[List[np.ndarray]] = None,
        mode_override: Optional[str] = None,
    ) -> None:
        mode = str(
            mode_override or getattr(args, "probe_cls_mode", "gap0")
        ).strip().lower()
        if mode == "knn_ensemble":
            if bank_list is None:
                raise ValueError("bank_list required for knn_ensemble probe mode")
            bank = self._probe_cls_bank(outputs, args)
            bank_np = bank.detach().cpu().numpy()
            bank_list.append(bank_np)
            # Keep CLS#0 features for patch-combo / t-SNE side paths.
            embed_list.append(bank_np[:, 0])
            return
        emb = self._probe_cls_embedding(
            outputs, args, seq_len=seq_len, mode_override=mode_override
        )
        embed_list.append(emb.detach().cpu().numpy())

    def _eval_knn_cls_bank_ensemble(
        self,
        X_bank: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        idx_train: np.ndarray,
        idx_test: np.ndarray,
        n_neighbors: int,
        task_name: str = "CLS bank ensemble",
        average_method: str = "macro",
        knn_weights: str = "uniform",
        embedding_preprocess: Optional[str] = None,
    ) -> float:
        """
        Fit one KNN per CLS head, then majority-vote over test predictions.
        X_bank: [N, K, D]
        """
        X_bank = np.asarray(X_bank, dtype=np.float32)
        n_heads = int(X_bank.shape[1])
        prep = ExpProbe._resolved_knn_emb_preprocess(embedding_preprocess, task_name)
        max_jobs = min(8, os.cpu_count() or 4)

        pred_stack = []
        for head_id in range(n_heads):
            X_tr = X_bank[idx_train, head_id]
            X_te = X_bank[idx_test, head_id]
            if prep != "none":
                X_tr, X_te = ExpProbe._preprocess_emb_for_euclidean_knn(X_tr, X_te, prep)
            knn = KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=knn_weights, n_jobs=max_jobs
            )
            knn.fit(X_tr, y_train)
            pred_stack.append(knn.predict(X_te))

        pred_mat = np.stack(pred_stack, axis=0)
        y_pred = np.zeros(pred_mat.shape[1], dtype=pred_mat.dtype)
        for i in range(pred_mat.shape[1]):
            vals, counts = np.unique(pred_mat[:, i], return_counts=True)
            y_pred[i] = vals[int(np.argmax(counts))]

        acc = accuracy_score(y_test, y_pred)
        b_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=average_method, zero_division=0)
        suffix = f" [prep={prep}]" if prep != "none" else ""
        print(f">>> KNN ({task_name}, {n_heads}-head vote){suffix}:")
        print(
            f"    Acc: {acc:.4f} | Balanced Acc: {b_acc:.4f} | "
            f"F1-{average_method}: {f1:.4f}"
        )
        return acc

    def _eval_probe_encoded_cls(
        self,
        X_embed: np.ndarray,
        y_all: np.ndarray,
        idx_train: np.ndarray,
        idx_test: np.ndarray,
        args: Any,
        n_neighbors: int,
        X_bank: Optional[np.ndarray] = None,
        task_name: str = "Encoded CLS",
    ) -> float:
        mode = str(getattr(args, "probe_cls_mode", "gap0")).strip().lower()
        if mode == "knn_ensemble":
            if X_bank is None:
                raise ValueError("X_bank required for knn_ensemble probe mode")
            return self._eval_knn_cls_bank_ensemble(
                X_bank,
                y_all[idx_train],
                y_all[idx_test],
                idx_train,
                idx_test,
                n_neighbors,
                task_name=f"{task_name} (ensemble)",
            )
        return self._eval_knn_metrics(
            X_embed[idx_train],
            y_all[idx_train],
            X_embed[idx_test],
            y_all[idx_test],
            n_neighbors,
            task_name,
            average_method="macro",
        )

    def _probe_cls_embedding(
        self,
        outputs: Dict[str, Any],
        args: Any,
        seq_len: Optional[int] = None,
        mode_override: Optional[str] = None,
    ) -> torch.Tensor:
        """Select CLS vector for downstream KNN (supports evidence-gap multi-CLS bank)."""
        return resolve_probe_cls_embedding(
            outputs, args, seq_len=seq_len, mode_override=mode_override
        )

    def _get_backbone(self, model: torch.nn.Module) -> torch.nn.Module:
        """Handle DataParallel and DDP wrapping safely."""
        # 处理 DDP (DistributedDataParallel)
        # DDP 模型有 module 属性，且类型名包含 'DistributedDataParallel'
        if hasattr(model, 'module'):
            model_type_name = type(model).__name__
            if 'DistributedDataParallel' in model_type_name or 'DDP' in model_type_name:
                return model.module
        # 处理 DataParallel
        if isinstance(model, torch.nn.DataParallel):
            return model.module
        return model

    @staticmethod
    def _preprocess_emb_for_euclidean_knn(
        X_train: np.ndarray,
        X_test: np.ndarray,
        mode: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """KNN(metric=euclidean) 之前的特征预处理；Scaler 仅用 train fit，避免泄漏。"""
        mode = mode or "none"
        Xt = np.asarray(X_train, dtype=np.float64)
        Xv = np.asarray(X_test, dtype=np.float64)
        if mode == "none":
            return Xt.astype(np.float32), Xv.astype(np.float32)
        if mode == "l2_rows":
            nt = np.linalg.norm(Xt, axis=1, keepdims=True)
            nt = np.maximum(nt, 1e-12)
            nv = np.linalg.norm(Xv, axis=1, keepdims=True)
            nv = np.maximum(nv, 1e-12)
            return (Xt / nt).astype(np.float32), (Xv / nv).astype(np.float32)
        if mode == "standardize":
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xt)
            Xva = scaler.transform(Xv)
            return Xtr.astype(np.float32), Xva.astype(np.float32)
        if mode == "l2_standardize":
            Xt_u, Xv_u = ExpProbe._preprocess_emb_for_euclidean_knn(X_train, X_test, "l2_rows")
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xt_u.astype(np.float64))
            Xva = scaler.transform(Xv_u.astype(np.float64))
            return Xtr.astype(np.float32), Xva.astype(np.float32)
        raise ValueError(f"Unknown embedding_preprocess: {mode!r}")

    @staticmethod
    def _resolved_knn_emb_preprocess(
        embedding_preprocess: Optional[str], _task_name: str
    ) -> str:
        """
        默认不进行预处理（与历史训练 log raw 欧氏 KNN probe 对齐，便于逐项对比）。
        显式传入 l2_rows / standardize / l2_standardize 时再启用；
        embedding_preprocess=None / '' / 'auto' 等价于 'none'。
        （task_name 不参与默认推断）
        """
        if embedding_preprocess in (None, "", "auto"):
            return "none"
        return embedding_preprocess

    def _eval_knn_metrics(self, X_train, y_train, X_test, y_test,
                          n_neighbors: int, task_name: str,
                          average_method: str = 'macro',
                          knn_weights: str = 'uniform',
                          embedding_preprocess: Optional[str] = None) -> float:
        """
        Unified KNN training and evaluation helper.

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签
            n_neighbors: KNN 的 k 值
            task_name: 任务名称
            average_method: F1 分数的平均方法
            knn_weights: 'uniform' 或 'distance'，不平衡数据可试 distance
            embedding_preprocess: 默认 None/auto → 与历史一致不做预处理；
                设为 l2_rows / standardize / l2_standardize 可再做对比实验。

        Returns:
            准确率分数
        """
        prep = ExpProbe._resolved_knn_emb_preprocess(embedding_preprocess, task_name)
        if prep != "none":
            X_train, X_test = ExpProbe._preprocess_emb_for_euclidean_knn(
                X_train, X_test, prep
            )
        # 限制线程数以避免与 OpenBLAS 冲突
        max_jobs = min(8, os.cpu_count() or 4)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=knn_weights, n_jobs=max_jobs)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        b_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=average_method, zero_division=0)

        # Try calculating AUC
        auc_score = "N/A"
        try:
            # Only calculate AUC for binary or if explicitly needed
            if len(np.unique(y_test)) == 2:
                probs = knn.predict_proba(X_test)[:, 1]
                auc_val = roc_auc_score(y_test, probs)
                auc_score = f"{auc_val:.4f}"
            elif hasattr(knn, "predict_proba"):
                # Optional: Multi-class AUC (One-vs-Rest)
                probs = knn.predict_proba(X_test)
                auc_val = roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')
                auc_score = f"{auc_val:.4f}"
        except:
            pass

        suffix = ""
        if prep != "none":
            suffix = f" [prep={prep}]"
        print(f">>> KNN ({task_name}){suffix}:")
        print(f"    Acc: {acc:.4f} | Balanced Acc: {b_acc:.4f} | F1-{average_method}: {f1:.4f} | AUC: {auc_score}")
        
        return acc

    def _eval_knn_cls_plus_patch_avg(
        self,
        X_all_cls: np.ndarray,
        X_all_patch_avg: Optional[np.ndarray],
        idx_train: np.ndarray,
        idx_test: np.ndarray,
        y_all: np.ndarray,
        n_neighbors: int,
        task_title_prefix: str = "",
        average_method: str = "macro",
    ) -> None:
        """
        TED-only：CLS 与 patch token 时间均值拼接后再 KNN；并试一路 CLS/Patch 双 KNN 概率晚融合。
        """
        if X_all_patch_avg is None:
            print(">>> [TED CLS+Patch] patch_tokens missing; skip CLS+PatchAvg KNN variants.")
            return

        pfx = (task_title_prefix or "").strip()
        title_base = f"{pfx} " if pfx else ""

        train_blocks: List[np.ndarray] = []
        test_blocks: List[np.ndarray] = []
        for feat in (X_all_cls, X_all_patch_avg):
            scaler = StandardScaler()
            train_blocks.append(scaler.fit_transform(feat[idx_train]))
            test_blocks.append(scaler.transform(feat[idx_test]))
        X_tr_z = np.concatenate(train_blocks, axis=1)
        X_te_z = np.concatenate(test_blocks, axis=1)
        print(
            f">>> [TED CLS+Patch] {title_base}z-score block concat: "
            f"X_train={X_tr_z.shape}, X_test={X_te_z.shape}"
        )
        self._eval_knn_metrics(
            X_tr_z,
            y_all[idx_train],
            X_te_z,
            y_all[idx_test],
            n_neighbors,
            f"{title_base}CLS+PatchAvg (z-score)".strip(),
            average_method=average_method,
        )

        def _l2_rows(X: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(X, axis=1, keepdims=True)
            norm = np.where(norm < 1e-8, 1.0, norm)
            return X / norm

        X_tr_l2 = np.concatenate(
            [_l2_rows(X_all_cls[idx_train]), _l2_rows(X_all_patch_avg[idx_train])],
            axis=1,
        )
        X_te_l2 = np.concatenate(
            [_l2_rows(X_all_cls[idx_test]), _l2_rows(X_all_patch_avg[idx_test])],
            axis=1,
        )
        print(
            f">>> [TED CLS+Patch] {title_base}L2-norm blocks concat: "
            f"X_train={X_tr_l2.shape}, X_test={X_te_l2.shape}"
        )
        self._eval_knn_metrics(
            X_tr_l2,
            y_all[idx_train],
            X_te_l2,
            y_all[idx_test],
            n_neighbors,
            f"{title_base}CLS+PatchAvg (L2-blocks)".strip(),
            average_method=average_method,
        )

        try:
            max_jobs = min(8, os.cpu_count() or 4)
            knn_c = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=max_jobs)
            knn_p = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=max_jobs)
            knn_c.fit(X_all_cls[idx_train], y_all[idx_train])
            knn_p.fit(X_all_patch_avg[idx_train], y_all[idx_train])
            proba_c = knn_c.predict_proba(X_all_cls[idx_test])
            proba_p = knn_p.predict_proba(X_all_patch_avg[idx_test])
            unified_classes = np.unique(y_all)

            def reindex_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
                cols = []
                for c in unified_classes:
                    idx = np.where(classes == c)[0]
                    cols.append(
                        proba[:, idx[0]] if len(idx) > 0 else np.zeros(proba.shape[0])
                    )
                return np.stack(cols, axis=1)

            proba_c = reindex_proba(proba_c, knn_c.classes_)
            proba_p = reindex_proba(proba_p, knn_p.classes_)
            y_te = y_all[idx_test]
            combined = 0.5 * proba_c + 0.5 * proba_p
            y_pred = np.argmax(combined, axis=1)
            acc_lf = accuracy_score(y_te, y_pred)
            b_acc_lf = balanced_accuracy_score(y_te, y_pred)
            f1_lf = f1_score(y_te, y_pred, average=average_method, zero_division=0)
            lf_name = f"{title_base}CLS+Patch LateFusion (0.5/0.5)".strip()
            print(f">>> KNN ({lf_name}):")
            print(
                f"    Acc: {acc_lf:.4f} | Balanced Acc: {b_acc_lf:.4f} | "
                f"F1-{average_method}: {f1_lf:.4f} | AUC: N/A"
            )
        except Exception as e_lf:
            print(f">>> [TED CLS+Patch] Late fusion failed: {e_lf}")

    @staticmethod
    def _segment_pool_by_years(tensor_data: torch.Tensor, num_segments: int) -> torch.Tensor:
        """
        按「年」对齐：将时间维 L 切分为 num_segments 段（对应每年一段），每段内做 mean 池化。
        能等分时严格等分；无法等分时使用 1 步重叠，保证每段长度一致且覆盖完整序列。

        Args:
            tensor_data: [B, L, D] 或 [B, L, C]
            num_segments: 年数（段数）

        Returns:
            [B, num_segments, D] 或 [B, num_segments, C]
        """
        B, L, D = tensor_data.shape
        if L <= 0 or num_segments <= 0:
            return tensor_data
        if num_segments == 1:
            return tensor_data.mean(dim=1, keepdim=True)
        base_len = L // num_segments
        remainder = L % num_segments
        if remainder == 0:
            # 能等分：无重叠
            out = []
            for i in range(num_segments):
                start = i * base_len
                end = start + base_len
                seg = tensor_data[:, start:end, :].mean(dim=1)
                out.append(seg)
            return torch.stack(out, dim=1)
        # 无法等分：每段长度 segment_len = ceil(L/num_segments)，段间重叠 1 步
        segment_len = math.ceil(L / num_segments)
        overlap = 1
        stride = max(1, segment_len - overlap)
        out = []
        for i in range(num_segments - 1):
            start = i * stride
            end = min(start + segment_len, L)
            seg = tensor_data[:, start:end, :].mean(dim=1)
            out.append(seg)
        # 最后一段对齐序列末尾，保证覆盖
        start_last = L - segment_len
        end_last = L
        seg_last = tensor_data[:, start_last:end_last, :].mean(dim=1)
        out.append(seg_last)
        return torch.stack(out, dim=1)

    @staticmethod
    def _interval_sse_linear_batch(pref: np.ndarray, l: int, r: int) -> np.ndarray:
        if r < l:
            return np.zeros((pref.shape[0],), dtype=np.float32)
        nseg = float(r - l + 1)
        s1 = nseg
        sx = pref[:, r + 1, 1] - pref[:, l, 1]
        sxx = pref[:, r + 1, 2] - pref[:, l, 2]
        sy = pref[:, r + 1, 3] - pref[:, l, 3]
        sxy = pref[:, r + 1, 4] - pref[:, l, 4]
        syy = pref[:, r + 1, 5] - pref[:, l, 5]

        den = np.maximum(s1 * sxx - sx * sx, 1e-8)
        a = (s1 * sxy - sx * sy) / den
        b = (sy - a * sx) / s1
        sse = syy - 2.0 * a * sxy - 2.0 * b * sy + a * a * sxx + 2.0 * a * b * sx + s1 * b * b
        return np.maximum(sse.astype(np.float32), 0.0)

    @staticmethod
    def _detect_one_break_batch(y: np.ndarray, min_seg: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        n, t = y.shape
        x = np.arange(t, dtype=np.float32)[None, :]
        x2 = x * x
        xy = x * y
        y2 = y * y

        pref = np.zeros((n, t + 1, 6), dtype=np.float64)
        pref[:, 1:, 1] = np.cumsum(np.broadcast_to(x, (n, t)), axis=1)
        pref[:, 1:, 2] = np.cumsum(np.broadcast_to(x2, (n, t)), axis=1)
        pref[:, 1:, 3] = np.cumsum(y, axis=1)
        pref[:, 1:, 4] = np.cumsum(xy, axis=1)
        pref[:, 1:, 5] = np.cumsum(y2, axis=1)

        total = ExpProbe._interval_sse_linear_batch(pref, 0, t - 1)
        best = np.full((n,), np.inf, dtype=np.float32)
        best_tau = np.full((n,), -1, dtype=np.int64)
        for tau in range(min_seg, t - min_seg):
            left = ExpProbe._interval_sse_linear_batch(pref, 0, tau - 1)
            right = ExpProbe._interval_sse_linear_batch(pref, tau, t - 1)
            cost = left + right
            m = cost < best
            best[m] = cost[m]
            best_tau[m] = tau
        impr = (np.maximum(total, 1e-8) - best) / np.maximum(total, 1e-8)
        return best_tau, np.nan_to_num(impr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def _smooth_cols(a: np.ndarray, width: int) -> np.ndarray:
        if width <= 1:
            return a.astype(np.float32)
        pad = width - 1
        p = np.pad(a, ((0, 0), (pad, 0)), mode='edge')
        o = np.zeros_like(a, dtype=np.float32)
        for k in range(width):
            o += p[:, k : k + a.shape[1]]
        return o / float(width)

    @staticmethod
    def _pca_project_topk(z_traj: np.ndarray, k: int = 3, fit_max_points: int = 200000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, w, d = z_traj.shape
        flat = z_traj.reshape(n * w, d).astype(np.float32, copy=False)
        valid = np.all(np.isfinite(flat), axis=1)
        flat = flat[valid]
        if flat.shape[0] == 0:
            return (
                np.zeros((n, w, k), dtype=np.float32),
                np.zeros((d,), dtype=np.float32),
                np.zeros((d, k), dtype=np.float32),
            )
        max_pts = int(max(1000, fit_max_points))
        if flat.shape[0] > max_pts:
            idx = np.linspace(0, flat.shape[0] - 1, num=max_pts, dtype=np.int64)
            fit = flat[idx]
        else:
            fit = flat
        mean_vec = fit.mean(axis=0).astype(np.float32)
        xc = fit - mean_vec[None, :]
        cov = (xc.T @ xc) / float(max(1, xc.shape[0] - 1))
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        order = np.argsort(eigvals)[::-1][:k]
        comps = eigvecs[:, order].astype(np.float32)
        comps = comps / (np.linalg.norm(comps, axis=0, keepdims=True) + 1e-8)
        zc = z_traj - mean_vec[None, None, :]
        proj = np.tensordot(zc, comps, axes=([-1], [0])).astype(np.float32)
        return proj, mean_vec, comps

    @staticmethod
    def _line_sse_prefix_multi(pref: np.ndarray, l: int, r: int, k: int) -> np.ndarray:
        if r < l:
            return np.zeros((pref.shape[0],), dtype=np.float32)
        nseg = float(r - l + 1)
        sx = pref[:, r + 1, 0] - pref[:, l, 0]
        sxx = pref[:, r + 1, 1] - pref[:, l, 1]
        den = np.maximum(nseg * sxx - sx * sx, 1e-8)
        sse_sum = np.zeros((pref.shape[0],), dtype=np.float32)
        for j in range(k):
            off = 2 + 3 * j
            sy = pref[:, r + 1, off + 0] - pref[:, l, off + 0]
            sxy = pref[:, r + 1, off + 1] - pref[:, l, off + 1]
            syy = pref[:, r + 1, off + 2] - pref[:, l, off + 2]
            a = (nseg * sxy - sx * sy) / den
            b = (sy - a * sx) / nseg
            sse = syy - 2.0 * a * sxy - 2.0 * b * sy + a * a * sxx + 2.0 * a * b * sx + nseg * b * b
            sse_sum += np.maximum(sse.astype(np.float32), 0.0)
        return sse_sum

    @staticmethod
    def _landtrend_break_on_multiseries(series: np.ndarray, min_seg: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        n, w, k = series.shape
        x = np.arange(w, dtype=np.float32)[None, :]
        x2 = x * x
        pref = np.zeros((n, w + 1, 2 + 3 * k), dtype=np.float64)
        pref[:, 1:, 0] = np.cumsum(np.broadcast_to(x, (n, w)), axis=1)
        pref[:, 1:, 1] = np.cumsum(np.broadcast_to(x2, (n, w)), axis=1)
        for j in range(k):
            y = series[:, :, j].astype(np.float32)
            off = 2 + 3 * j
            pref[:, 1:, off + 0] = np.cumsum(y, axis=1)
            pref[:, 1:, off + 1] = np.cumsum(x * y, axis=1)
            pref[:, 1:, off + 2] = np.cumsum(y * y, axis=1)
        total = ExpProbe._line_sse_prefix_multi(pref, 0, w - 1, k)
        best = np.full(n, np.inf, dtype=np.float32)
        best_tau = np.full(n, -1, dtype=np.int64)
        for tau in range(min_seg, w - min_seg):
            c = ExpProbe._line_sse_prefix_multi(pref, 0, tau - 1, k) + ExpProbe._line_sse_prefix_multi(pref, tau, w - 1, k)
            m = c < best
            best = np.where(m, c, best)
            best_tau = np.where(m, tau, best_tau)
        impr = (np.maximum(total, 1e-8) - best) / np.maximum(total, 1e-8)
        impr = np.nan_to_num(impr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return best_tau, impr

    @staticmethod
    def _tau_to_pred_step(best_tau: np.ndarray, starts: np.ndarray, win_len: int) -> np.ndarray:
        n = best_tau.shape[0]
        pred = np.full(n, -1, dtype=np.int64)
        wn = len(starts)
        for i in range(n):
            tau = int(best_tau[i])
            if tau < 1 or tau >= wn:
                continue
            pred[i] = int(round((starts[tau - 1] + win_len / 2.0 + starts[tau] + win_len / 2.0) / 2.0))
        return pred

    def _run_new_change_segmentation_probe(
        self,
        model: torch.nn.Module,
        args: Any,
        dataset_key: str,
        imputator=None,
    ) -> Dict[str, float]:
        ds_root = os.path.join(
            getattr(args, 'downstream_data_root', '/intelnvme01/ziyun/DownStreamTasks'),
            'downstream_segmentation_task',
            'hls_composite_nc',
        )
        file_map = {
            'hansen': 'segmentation_hansen_hls_processed.nc',
            'wildfire': 'segmentation_wildfire_hls_processed.nc',
            'lcmapchange': 'segmentation_lcmap_hls_processed.nc',
        }
        nc_path = os.path.join(ds_root, file_map[dataset_key])
        if not os.path.exists(nc_path):
            print(f">>> [SegProbe-{dataset_key}] missing dataset: {nc_path}")
            return {}

        with Dataset(nc_path, 'r') as f:
            data = np.asarray(f.variables['data'][:], dtype=np.float32)
            time_raw = np.asarray(f.variables['time'][:]).astype(str)
            dims = f.variables['data'].dimensions
            if len(dims) == 3 and dims[0] == 'time':
                x_all = np.transpose(data, (2, 0, 1))
            else:
                x_all = data
            if 'change_date_1' in f.variables:
                change_date_1 = np.asarray(f.variables['change_date_1'][:]).astype(str)
            else:
                change_date_1 = np.array([''] * x_all.shape[0], dtype=object)
            change_year_1 = np.asarray(f.variables['change_year_1'][:]) if 'change_year_1' in f.variables else None
            num_change_points = np.asarray(f.variables['num_change_points'][:]) if 'num_change_points' in f.variables else np.zeros((x_all.shape[0],), dtype=np.int64)
            if 'lon' in f.variables and 'lat' in f.variables:
                lon = np.asarray(f.variables['lon'][:], dtype=np.float32)
                lat = np.asarray(f.variables['lat'][:], dtype=np.float32)
            elif 'longitude' in f.variables and 'latitude' in f.variables:
                lon = np.asarray(f.variables['longitude'][:], dtype=np.float32)
                lat = np.asarray(f.variables['latitude'][:], dtype=np.float32)
            else:
                lon = None
                lat = None

        t_idx = pd.to_datetime(np.asarray(time_raw, dtype=str), format='%Y%j', errors='coerce')
        t_ord = np.asarray([z.toordinal() if pd.notna(z) else -1 for z in t_idx], dtype=np.int64)
        gt_step = np.full((x_all.shape[0],), -1, dtype=np.int64)
        valid_gt = np.zeros((x_all.shape[0],), dtype=bool)
        for i in range(x_all.shape[0]):
            ev = pd.NaT
            v = str(change_date_1[i]).strip()
            if v not in {'', 'nan', 'NaT'}:
                ev = pd.to_datetime(v, errors='coerce')
            if pd.isna(ev) and change_year_1 is not None and int(change_year_1[i]) > 0:
                ev = pd.to_datetime(f"{int(change_year_1[i])}-07-01", errors='coerce')
            if pd.notna(ev) and np.any(t_ord >= 0):
                eo = int(ev.toordinal())
                gt_step[i] = int(np.argmin(np.abs(t_ord - eo)))
                valid_gt[i] = True

        # Match old evaluation filtering to avoid impossible edge breakpoints.
        # Strictly match legacy eval_real_downstream_nc_change.py filtering.
        min_useful_gt = max(1, int(getattr(args, 'seg_sse_min_seg', 10))) * int(getattr(args, 'stride', 3)) + int(getattr(args, 'win_len', 12))
        valid = valid_gt & (gt_step >= min_useful_gt) & (gt_step < x_all.shape[1] - min_useful_gt)
        idxs = np.where(valid)[0]
        if idxs.size == 0:
            print(f">>> [SegProbe-{dataset_key}] no valid gt change points after margin filter.")
            return {}
        x_sub = x_all[idxs]
        gt_sub = gt_step[idxs]
        lon_sub = lon[idxs] if lon is not None else None
        lat_sub = lat[idxs] if lat is not None else None

        backbone_model = self._get_backbone(model)
        backbone_model.eval().to(self.device)

        # Match old eval preprocessing/timeMark generation.
        x_sub = applyPreScaling(x_sub, getDefaultPreScaler()).astype(np.float32, copy=False)
        tm_stamp = buildTimeMarkFromTime(np.asarray(time_raw, dtype=object), freq='RS', timeenc=1).astype(np.float32)
        tm = np.tile(tm_stamp[None, :, :], (x_sub.shape[0], 1, 1))

        patch_series = []
        bs = int(getattr(args, 'batch_size', 256))
        with torch.no_grad():
            for s in range(0, x_sub.shape[0], bs):
                e = min(s + bs, x_sub.shape[0])
                bx = torch.from_numpy(x_sub[s:e]).to(self.device, dtype=torch.float32)
                btm = torch.from_numpy(tm[s:e]).to(self.device, dtype=torch.float32)
                # Legacy script does not use lon_lat or AMP here.
                _ = lon_sub, lat_sub
                out = backbone_model.encode(bx, btm, imputator=imputator, lon_lat=None)
                pt = out.get('patch_tokens', None)
                if pt is None:
                    print(f">>> [SegProbe-{dataset_key}] patch_tokens missing, skip.")
                    return {}
                if pt.ndim == 4:
                    pt = pt.mean(dim=1)
                patch_series.append(pt.detach().float().cpu().numpy())

        patch_series = np.concatenate(patch_series, axis=0)
        n, p, d = patch_series.shape
        patch_series = np.nan_to_num(patch_series, nan=0.0, posinf=0.0, neginf=0.0)
        patch_series = patch_series / (np.linalg.norm(patch_series, axis=-1, keepdims=True) + 1e-8)

        pca3, _, _ = self._pca_project_topk(
            patch_series,
            k=3,
            fit_max_points=int(getattr(args, 'seg_pca_fit_max_points', 120000)),
        )
        smooth_w = int(getattr(args, 'seg_pca_smooth_width', 3))
        if smooth_w > 1:
            for kk in range(pca3.shape[-1]):
                pca3[:, :, kk] = self._smooth_cols(pca3[:, :, kk], smooth_w)
        pred_patch, score = self._landtrend_break_on_multiseries(
            pca3,
            min_seg=max(3, int(getattr(args, 'seg_sse_min_seg', 10))),
        )

        patch_len = int(getattr(args, 'patch_len', 3))
        patch_stride = int(getattr(args, 'stride', 3))
        starts_patch = np.arange(0, max(1, x_sub.shape[1] - patch_len + 1), patch_stride, dtype=np.int64)
        last_start = max(0, x_sub.shape[1] - patch_len)
        if starts_patch.size == 0 or starts_patch[-1] != last_start:
            starts_patch = np.concatenate([starts_patch, np.array([last_start], dtype=np.int64)])
        # Keep timeline length aligned with patch token count from model.encode.
        if starts_patch.shape[0] != p:
            starts_patch = np.linspace(0, max(0, x_sub.shape[1] - patch_len), num=p, dtype=np.int64)
        pred_step = self._tau_to_pred_step(pred_patch, starts_patch, patch_len)
        pred_step = np.clip(pred_step, 0, x_sub.shape[1] - 1).astype(np.int64)

        # Match historical "msrefine15": multiscale CLS evidence around coarse patch breakpoint.
        refine_k = int(getattr(args, 'seg_local_refine_k', 15))
        if refine_k > 0:
            ws = [10, 30, 60]
            ww = np.asarray([0.3, 0.4, 0.3], dtype=np.float32)
            stride_ms = int(getattr(args, 'seg_multiscale_stride', 3))
            ms_pack = []
            with torch.no_grad():
                for wlen in ws:
                    starts_ms = np.arange(0, max(1, x_sub.shape[1] - wlen + 1), stride_ms, dtype=np.int64)
                    if starts_ms.size == 0 or starts_ms[-1] != x_sub.shape[1] - wlen:
                        starts_ms = np.concatenate([starts_ms, np.array([max(0, x_sub.shape[1] - wlen)], dtype=np.int64)])
                    centers_ms = starts_ms.astype(np.float32) + float(wlen) / 2.0
                    z_ms = np.zeros((x_sub.shape[0], starts_ms.size, int(backbone_model.backbone.d_model)), dtype=np.float32)
                    for wi, s0 in enumerate(starts_ms):
                        e0 = int(s0 + wlen)
                        for s in range(0, x_sub.shape[0], bs):
                            e = min(s + bs, x_sub.shape[0])
                            bx = torch.from_numpy(x_sub[s:e, s0:e0, :]).to(self.device, dtype=torch.float32)
                            btm = torch.from_numpy(tm[s:e, s0:e0, :]).to(self.device, dtype=torch.float32)
                            out = backbone_model.encode(bx, btm, imputator=imputator, lon_lat=None)
                            cls = self._probe_cls_embedding(
                                out, args, seq_len=int(bx.shape[1]), mode_override="auto_gap"
                            ).detach().float().cpu().numpy()
                            z_ms[s:e, wi, :] = cls
                    z_ms = z_ms / (np.linalg.norm(z_ms, axis=-1, keepdims=True) + 1e-8)
                    ms_pack.append((wlen, z_ms, centers_ms))

            refined = pred_step.copy()
            tmax = x_sub.shape[1] - 1
            min_improve = float(getattr(args, 'seg_multiscale_min_improve', 0.0))
            for i in range(refined.shape[0]):
                lo = max(0, int(refined[i]) - refine_k)
                hi = min(tmax, int(refined[i]) + refine_k)
                if hi <= lo:
                    continue
                cands = np.arange(lo, hi + 1, dtype=np.int64)
                score_local = np.zeros((cands.shape[0],), dtype=np.float32)
                for wi, (wlen, z_ms, centers_ms) in enumerate(ms_pack):
                    half = float(wlen) / 2.0
                    pre_t = cands.astype(np.float32) - half
                    post_t = cands.astype(np.float32) + half
                    idx_pre = np.clip(np.searchsorted(centers_ms, pre_t), 0, len(centers_ms) - 1)
                    idx_post = np.clip(np.searchsorted(centers_ms, post_t), 0, len(centers_ms) - 1)
                    a = z_ms[i, idx_pre, :]
                    b = z_ms[i, idx_post, :]
                    cs = np.sum(a * b, axis=-1).clip(-1.0, 1.0)
                    score_local += float(ww[wi]) * (1.0 - cs).astype(np.float32)
                best_j = int(np.argmax(score_local))
                old_j = int(np.clip(int(refined[i]) - lo, 0, cands.shape[0] - 1))
                if score_local[best_j] >= score_local[old_j] + min_improve:
                    refined[i] = int(cands[best_j])
            pred_step = refined
        err = np.abs(pred_step.astype(np.int64) - gt_sub.astype(np.int64)).astype(np.float32)
        change_bin = (num_change_points[idxs] > 0).astype(np.int64)
        pred_bin = (pred_patch >= 0).astype(np.int64)
        bin_acc = float((pred_bin == change_bin).mean())

        print(
            f">>> [SegProbe-{dataset_key}] N={x_all.shape[0]} used={int(idxs.size)} "
            f"MAE={float(err.mean()):.2f} MedAE={float(np.median(err)):.2f} "
            f"Hit@15={float((err <= 15).mean()):.3f} BinAcc={bin_acc:.3f} "
            f"PCAk=3 Smooth={smooth_w} RefineK={refine_k} ScoreMean={float(score.mean()):.3f}"
        )
        return {
            f"{dataset_key}_mae_steps": float(err.mean()),
            f"{dataset_key}_hit_at_15": float((err <= 15).mean()),
            f"{dataset_key}_bin_acc": bin_acc,
        }

    def knn_probe_Hansen_Segmentation(self, model: torch.nn.Module, args: Any, n_neighbors: int = 5, imputator=None) -> Dict[str, float]:
        print(">>> [ExpProbe] Starting Hansen change-point probe (patch+pca+landtrend)...")
        return self._run_new_change_segmentation_probe(model, args, 'hansen', imputator=imputator)

    def knn_probe_Wildfire_Segmentation(self, model: torch.nn.Module, args: Any, n_neighbors: int = 5, imputator=None) -> Dict[str, float]:
        print(">>> [ExpProbe] Starting Wildfire change-point probe (patch+pca+landtrend)...")
        return self._run_new_change_segmentation_probe(model, args, 'wildfire', imputator=imputator)

    def knn_probe_LCMAPChange_Segmentation(self, model: torch.nn.Module, args: Any, n_neighbors: int = 5, imputator=None) -> Dict[str, float]:
        print(">>> [ExpProbe] Starting LCMAP-change probe (single-break, first cp)...")
        return self._run_new_change_segmentation_probe(model, args, 'lcmapchange', imputator=imputator)

    def knn_probe_LCMAP_Classification(self, model: torch.nn.Module, args: Any, n_neighbors: int = 5, imputator=None) -> Dict[str, float]:
        print(f">>> [ExpProbe] Starting KNN LCMAP Classification (k={n_neighbors})...")
        probe_cls_mode = str(getattr(args, "probe_cls_mode", "gap0"))
        print(f">>> [ProbeCLS] mode={probe_cls_mode} tau={getattr(args, 'probe_cls_fusion_tau', 1.0)}")
        cloud_mask_ratio = float(getattr(args, "probe_cloud_mask_ratio", 0.0))
        cloud_mask_seed = int(getattr(args, "probe_cloud_mask_seed", 2026))
        if cloud_mask_ratio > 0:
            print(f">>> [CloudMask] ratio={cloud_mask_ratio} (seed={cloud_mask_seed})")
        
        # 生成缓存键（基于数据集标识）
        cls_min_samples = int(getattr(args, 'cls_min_samples_per_class', 10))
        cls_few_shot_k = getattr(args, 'cls_few_shot_k', None)
        cache_key = (
            f"LCMAP_Classification_{args.data}_{args.seq_len}_{args.sampling_stride if hasattr(args, 'sampling_stride') else 'default'}"
            f"_min{cls_min_samples}_few{cls_few_shot_k or 'ratio'}"
        )
        # 检查是否有缓存的baseline数据
        use_cached_baseline = cache_key in _baseline_cache
        
        # 关键修复：KNN Probe只在rank 0运行，需要禁用DDP切分以获取全部数据
        _, data_loader = data_provider_LCMAP_Classification(args, flag='LCMAP_Classification', disable_ddp_split=True)
        print('args.num workers', args.num_workers)
        backbone_model = self._get_backbone(model)
        backbone_model.eval()
        backbone_model.to(self.device)

        preds_embed_list, preds_patch_avg_list, labels_list = [], [], []
        preds_bank_list = [] if probe_cls_mode == "knn_ensemble" else None
        
        # 只在第一次或需要重新计算baseline时提取原始特征
        if not use_cached_baseline:
            preds_raw_list = []

        with torch.no_grad():
            for step, batch in enumerate(data_loader):
                batch_x = batch[0].float().to(self.device)
                batch_x_mark = batch[1].float().to(self.device)
                labels = batch[2].to(self.device)

                # simulate clouds BEFORE encode (and before raw baselines)
                batch_x = self._apply_probe_cloud_mask(batch_x, cloud_mask_ratio, cloud_mask_seed, step)

                with autocast():
                    outputs = backbone_model.encode(batch_x, batch_x_mark, imputator=imputator)

                # 1. Model Feature (CLS Token / CLS bank)
                self._append_probe_cls_batch(
                    outputs,
                    args,
                    int(batch_x.shape[1]),
                    preds_embed_list,
                    preds_bank_list,
                )

                labels_list.append(labels.detach().cpu().numpy())

                # 2. Model Feature (Patch Tokens Avg) - 可选
                patch_tokens = outputs.get('patch_tokens', None)
                if patch_tokens is not None:
                    # 期望形状 [B, T, D]
                    if patch_tokens.ndim == 3:
                        patch_avg = patch_tokens.mean(dim=1)
                    elif patch_tokens.ndim == 4:
                        # [B, heads, T, D] 或类似结构，先在 head 维上做平均
                        patch_avg = patch_tokens.mean(dim=1).mean(dim=1)
                    else:
                        patch_avg = None
                    if patch_avg is not None:
                        preds_patch_avg_list.append(patch_avg.detach().cpu().numpy())
                
                # 3. Raw Feature (Mean over time) - 只在第一次计算
                if not use_cached_baseline:
                    batch_x_filled = torch.nan_to_num(batch_x, nan=0.0)
                    raw_feat = batch_x_filled.mean(dim=1)
                    preds_raw_list.append(raw_feat.detach().cpu().numpy())

        # ==========================================
        # 1. 处理 Embedding (模型特征) - 移除窗口合并逻辑
        # ==========================================
        X_all_embed = np.concatenate(preds_embed_list, axis=0) 
        y_all = np.concatenate(labels_list, axis=0).reshape(-1)
        X_all_bank = None
        if preds_bank_list is not None and len(preds_bank_list) > 0:
            X_all_bank = np.concatenate(preds_bank_list, axis=0)
        del preds_embed_list, labels_list, preds_bank_list

        X_all_patch_avg = None
        if len(preds_patch_avg_list) > 0:
            try:
                X_all_patch_avg = np.concatenate(preds_patch_avg_list, axis=0)
                print(f">>> Patch-avg features: X={X_all_patch_avg.shape}")
            except Exception as e:
                print(f">>> [Warning] Failed to concatenate patch-avg features, will skip patch-based combinations. Error: {e}")
                X_all_patch_avg = None
        preds_patch_avg_list = []  # 及时释放

        # 移除窗口合并逻辑：每个序列都是独立样本
        print(f">>> Embedding features: X={X_all_embed.shape}, y={y_all.shape}")
        if X_all_bank is not None:
            print(f">>> CLS bank features: X={X_all_bank.shape}")
        y_all_before_class_filter = y_all.copy()

        # 排除「类别总样本数 < cls_min_samples_per_class」的类别
        idx_keep, y_remap = _filter_small_classes(y_all, min_samples_per_class=cls_min_samples)
        if len(idx_keep) == 0:
            print(f">>> [LCMAP] No samples left after filtering classes with < {cls_min_samples} samples. Skip.")
            return {}
        X_all_embed = X_all_embed[idx_keep]
        y_all = y_remap
        if X_all_bank is not None:
            X_all_bank = X_all_bank[idx_keep]
        if X_all_patch_avg is not None:
            X_all_patch_avg = X_all_patch_avg[idx_keep]
        print(f">>> [LCMAP] After filtering small classes: N={len(y_all)}, n_class={len(np.unique(y_all))}")

        # 若需要 t-SNE，可在这里保存 TED 的 CLS 表达（全样本中按类采样）
        if getattr(args, 'save_tsne_embeddings', False):
            try:
                from pathlib import Path
                out_dir = Path("logs")
                out_dir.mkdir(exist_ok=True)
                X_tsne, y_tsne = _sample_per_class_np(X_all_embed, y_all, max_per_class=20, seed=2026)
                np.savez(out_dir / "tsne_LCMAP_TED_embeddings.npz", X=X_tsne, y=y_tsne)
                print(f">>> [t-SNE] Saved LCMAP TED embeddings for t-SNE: {out_dir / 'tsne_LCMAP_TED_embeddings.npz'}")
            except Exception as e:
                print(f">>> [t-SNE Warning] Failed to save LCMAP TED embeddings: {e}")

        # ==========================================
        # 2. 处理 Raw Feature (原始特征) - 移除窗口合并逻辑
        # ==========================================
        if use_cached_baseline:
            # A. 从缓存读取
            cached_data = _baseline_cache[cache_key]
            X_all_raw = cached_data['X_all_raw'] # 原始特征 (N, C)
            idx_train = cached_data['idx_train']
            idx_test = cached_data['idx_test']
            print(f">>> [Using cached baseline] Train: {len(idx_train)}, Test: {len(idx_test)}")
        else:
            # B. 首次计算：拼接 + 过滤后缓存（X_all_raw 已在上面用 idx_keep 过滤过，此处用当前 y_all 长度一致）
            X_all_raw = np.concatenate(preds_raw_list, axis=0)
            del preds_raw_list
            X_all_raw = X_all_raw[idx_keep]

            # 生成 Split：few-shot 或 比例
            cls_seed = int(getattr(args, 'cls_split_seed', 42))
            cls_ratio = float(getattr(args, 'cls_train_ratio', 0.8))
            if cls_few_shot_k is not None and cls_few_shot_k > 0:
                idx_train, idx_test = _get_few_shot_split(len(y_all), y_all, cls_few_shot_k, cls_seed)
                print(f">>> [Split] Few-shot k={cls_few_shot_k} per class, seed={cls_seed}")
            else:
                indices = np.arange(len(y_all))
                try:
                    idx_train, idx_test = train_test_split(
                        indices, train_size=cls_ratio, random_state=cls_seed, stratify=y_all
                    )
                except ValueError:
                    idx_train, idx_test = train_test_split(indices, train_size=cls_ratio, random_state=cls_seed)
                print(f">>> [Split] train_ratio={cls_ratio}, seed={cls_seed}")

            _baseline_cache[cache_key] = {
                'X_all_raw': X_all_raw,
                'idx_train': idx_train,
                'idx_test': idx_test
            }
            print(f">>> [Computing baseline] Train: {len(idx_train)}, Test: {len(idx_test)}")

        # Print Distribution
        unique, counts = np.unique(y_all, return_counts=True)
        print(f">>> Class Distribution: {dict(zip(unique, counts))}")

        # ==========================================
        # 3. 评估 (Eval) - 基础结果
        # ==========================================
        
        # 评估 Model Feature
        self._eval_probe_encoded_cls(
            X_all_embed,
            y_all,
            idx_train,
            idx_test,
            args,
            n_neighbors,
            X_bank=X_all_bank,
            task_name="Encoded CLS",
        )

        self._eval_knn_cls_plus_patch_avg(
            X_all_embed,
            X_all_patch_avg,
            idx_train,
            idx_test,
            y_all,
            n_neighbors,
            task_title_prefix="",
        )

        # 评估 Raw Feature (Baseline)
        if use_cached_baseline:
            cached_result = _baseline_cache[cache_key].get('baseline_result_classification')
            if cached_result:
                print(cached_result, end='')
            else:
                # 理论上不该进这里，除非缓存结构不完整，但也补一个计算
                 pass 
        else:
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            # X_all_raw 形状: (N, C)，和 idx 对应
            self._eval_knn_metrics(X_all_raw[idx_train], y_all[idx_train], 
                                   X_all_raw[idx_test], y_all[idx_test], 
                                   n_neighbors, "Raw Mean", average_method='macro')
            baseline_result = buffer.getvalue()
            sys.stdout = old_stdout
            print(baseline_result, end='')
            _baseline_cache[cache_key]['baseline_result_classification'] = baseline_result

        # ==========================================
        # 4. 可选：AlphaEarth Embedding 分析
        # ==========================================
        use_alphaearth = getattr(args, 'use_alphaearth', False)
        if use_alphaearth:
            print("\n>>> [AlphaEarth] Start AlphaEarth embeddings analysis for LCMAP Classification...")
            alphaearth_path = getattr(args, 'alphaearth_path', None)
            if not alphaearth_path:
                _ds_root = getattr(args, 'downstream_data_root', '/intelnvme01/ziyun/DownStreamTasks')
                alphaearth_path = _resolve_first_existing_path([
                    os.path.join(_ds_root, 'downstream_classification_task', 'lcmap_gse_classification.npz'),
                    os.path.join(_ds_root, 'downstream_classification_task', 'lcmap_alphaearth_classification.npz'),
                    os.path.join(_ds_root, 'LCMAP&Alphaearth', 'ae_classification_dataset.npz'),
                ])
            if not alphaearth_path:
                print(
                    ">>> [AlphaEarth Warning] LCMAP AlphaEarth file not found. "
                    "Tried lcmap_gse_classification.npz / lcmap_alphaearth_classification.npz / "
                    "legacy ae_classification_dataset.npz."
                )
                alphaearth_path = "__MISSING_ALPHAEARTH_FILE__.npz"

            try:
                ae_npz = np.load(alphaearth_path, allow_pickle=True)
                ae_data = ae_npz['data']    # (N_ae, T_years, 64)
                ae_data = _validate_alphaearth_array(ae_data, "LCMAP AlphaEarth")
                ae_labels = ae_npz['labels']
                ae_time = ae_npz['time']    # (T_years,)

                # 与 TED 一致：按小类过滤后的 idx_keep 取 AlphaEarth 行（不可截断前 N 行）
                ae_data = _alphaearth_data_after_class_filter(
                    ae_data, idx_keep, "LCMAP Classification + AlphaEarth"
                )
                y_all_alpha = y_all
                X_all_embed_alpha = X_all_embed
                X_all_raw_alpha = X_all_raw
                idx_train_alpha = idx_train
                idx_test_alpha = idx_test
                if X_all_patch_avg is not None:
                    X_all_patch_alpha = X_all_patch_avg
                else:
                    X_all_patch_alpha = None

                # 诊断：npz 标签应与过滤前 HLS 标签在相同 idx 上一致（TED 的 y 已 remap，勿拿 ae_labels 与 y_all 直接比）
                try:
                    ae_lab = np.asarray(ae_labels).reshape(-1)
                    if ae_lab.size > int(idx_keep.max()):
                        y_exp = y_all_before_class_filter[idx_keep]
                        if not np.array_equal(ae_lab[idx_keep].astype(y_exp.dtype, copy=False), y_exp):
                            print(
                                ">>> [AlphaEarth Warning] LCMAP: ae_labels[idx_keep] != HLS labels before filter "
                                "for kept rows; check npz vs dataloader order."
                            )
                except Exception:
                    pass

                # 统一策略：AlphaEarth 代表空间平均状态，如存在多份时间/年份信息，直接在时间维度做平均。
                # 若 data 为 3 维 (N, T, D)，在 T 维上取均值得到 (N, D)；若本身已是 2 维 (N, D)，直接使用。
                if ae_data.ndim == 3:
                    ae_avg = ae_data.mean(axis=1).astype(np.float32)
                elif ae_data.ndim == 2:
                    ae_avg = ae_data.astype(np.float32)
                else:
                    raise ValueError(f"Unexpected LCMAP AlphaEarth data ndim={ae_data.ndim}, expected 2 or 3.")
                print(f">>> [AlphaEarth] LCMAP AlphaEarth ae_avg shape = {ae_avg.shape}, time_len={ae_data.shape[1] if ae_data.ndim == 3 else 1}")

                # 若需要 t-SNE，同样保存 AlphaEarth 的 ae_avg 表达
                if getattr(args, 'save_tsne_embeddings', False):
                    try:
                        from pathlib import Path
                        out_dir = Path("logs")
                        out_dir.mkdir(exist_ok=True)
                        X_tsne_ae, y_tsne_ae = _sample_per_class_np(ae_avg, y_all_alpha, max_per_class=20, seed=2026)
                        np.savez(out_dir / "tsne_LCMAP_AlphaEarth_embeddings.npz", X=X_tsne_ae, y=y_tsne_ae)
                        print(f">>> [t-SNE] Saved LCMAP AlphaEarth embeddings for t-SNE: {out_dir / 'tsne_LCMAP_AlphaEarth_embeddings.npz'}")
                    except Exception as e:
                        print(f">>> [t-SNE Warning] Failed to save LCMAP AlphaEarth embeddings: {e}")

                # 特征拼接前进行简单的特征尺度对齐：对每个特征做 z-score（基于训练集）
                from sklearn.preprocessing import StandardScaler

                def _l2_norm_rows(X: np.ndarray) -> np.ndarray:
                    """按行 L2 归一化，避免某一块主导距离。"""
                    norm = np.linalg.norm(X, axis=1, keepdims=True)
                    norm = np.where(norm < 1e-8, 1.0, norm)
                    return X / norm

                def build_concat_features(train_idx, test_idx, feat_list, name: str):
                    """对每个特征块做标准化之后拼接。"""
                    train_blocks = []
                    test_blocks = []
                    for feat in feat_list:
                        if feat is None:
                            continue
                        scaler = StandardScaler()
                        feat_train = scaler.fit_transform(feat[train_idx])
                        feat_test = scaler.transform(feat[test_idx])
                        train_blocks.append(feat_train)
                        test_blocks.append(feat_test)
                    if not train_blocks:
                        return None, None
                    X_train_cat = np.concatenate(train_blocks, axis=1)
                    X_test_cat = np.concatenate(test_blocks, axis=1)
                    print(f">>> [AlphaEarth] {name} features: X_train={X_train_cat.shape}, X_test={X_test_cat.shape}")
                    return X_train_cat, X_test_cat

                def build_concat_l2_blocks(train_idx, test_idx, feat_list, name: str):
                    """对每个特征块按行 L2 归一化后再拼接，使两路在距离中权重相当。"""
                    train_blocks = []
                    test_blocks = []
                    for feat in feat_list:
                        if feat is None:
                            continue
                        train_blocks.append(_l2_norm_rows(feat[train_idx]))
                        test_blocks.append(_l2_norm_rows(feat[test_idx]))
                    if not train_blocks:
                        return None, None
                    X_train_cat = np.concatenate(train_blocks, axis=1)
                    X_test_cat = np.concatenate(test_blocks, axis=1)
                    print(f">>> [AlphaEarth] {name} (L2-norm blocks) features: X_train={X_train_cat.shape}, X_test={X_test_cat.shape}")
                    return X_train_cat, X_test_cat

                # 1) 仅 CLS（已在基础结果中评估，这里仅做说明，不重复计算）
                print(">>> [AlphaEarth] Baseline Encoded CLS accuracy is reported above (without AlphaEarth).")

                # 1.5) 仅 AlphaEarth Avg
                X_train_ae_only, X_test_ae_only = build_concat_features(
                    idx_train_alpha, idx_test_alpha,
                    [ae_avg],
                    name="AlphaEarth Only"
                )
                if X_train_ae_only is not None:
                    self._eval_knn_metrics(
                        X_train_ae_only, y_all_alpha[idx_train_alpha],
                        X_test_ae_only, y_all_alpha[idx_test_alpha],
                        n_neighbors, "AlphaEarthOnly", average_method='macro'
                    )
                    # 不平衡时 distance 加权有时更好
                    self._eval_knn_metrics(
                        X_train_ae_only, y_all_alpha[idx_train_alpha],
                        X_test_ae_only, y_all_alpha[idx_test_alpha],
                        n_neighbors, "AlphaEarthOnly (weights=distance)", average_method='macro', knn_weights='distance'
                    )

                # 2) AlphaEarth Avg + CLS（z-score 拼接，可能被某一维主导）
                X_train_ae_cls, X_test_ae_cls = build_concat_features(
                    idx_train_alpha, idx_test_alpha,
                    [ae_avg, X_all_embed_alpha],
                    name="AlphaEarth + CLS"
                )
                if X_train_ae_cls is not None:
                    self._eval_knn_metrics(
                        X_train_ae_cls, y_all_alpha[idx_train_alpha],
                        X_test_ae_cls, y_all_alpha[idx_test_alpha],
                        n_neighbors, "AlphaEarth+CLS", average_method='macro'
                    )

                # 2b) AlphaEarth + CLS（按块 L2 归一化再拼接，两路权重相当，有望提升联合精度）
                X_train_ae_cls_l2, X_test_ae_cls_l2 = build_concat_l2_blocks(
                    idx_train_alpha, idx_test_alpha,
                    [ae_avg, X_all_embed_alpha],
                    name="AlphaEarth + CLS"
                )
                if X_train_ae_cls_l2 is not None:
                    self._eval_knn_metrics(
                        X_train_ae_cls_l2, y_all_alpha[idx_train_alpha],
                        X_test_ae_cls_l2, y_all_alpha[idx_test_alpha],
                        n_neighbors, "AlphaEarth+CLS (L2-norm)", average_method='macro'
                    )
                    self._eval_knn_metrics(
                        X_train_ae_cls_l2, y_all_alpha[idx_train_alpha],
                        X_test_ae_cls_l2, y_all_alpha[idx_test_alpha],
                        n_neighbors, "AlphaEarth+CLS (L2-norm, weights=distance)", average_method='macro', knn_weights='distance'
                    )

                # 2c) 晚融合：双路 KNN 概率融合。简单平均易被“错路”拉低，改为加权平均（更强一路权重大）
                try:
                    max_jobs = min(8, os.cpu_count() or 4)
                    knn_ae = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=max_jobs)
                    knn_cls = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=max_jobs)
                    knn_ae.fit(X_train_ae_only, y_all_alpha[idx_train_alpha])
                    knn_cls.fit(X_all_embed_alpha[idx_train_alpha], y_all_alpha[idx_train_alpha])
                    proba_ae = knn_ae.predict_proba(X_test_ae_only)
                    proba_cls = knn_cls.predict_proba(X_all_embed_alpha[idx_test_alpha])
                    # 按统一类别顺序重排两路概率列，避免两路 classes_ 顺序或缺类不一致
                    unified_classes = np.unique(y_all_alpha)

                    def reindex_proba(proba, classes):
                        cols = []
                        for c in unified_classes:
                            idx = np.where(classes == c)[0]
                            cols.append(proba[:, idx[0]] if len(idx) > 0 else np.zeros(proba.shape[0]))
                        return np.stack(cols, axis=1)

                    proba_ae = reindex_proba(proba_ae, knn_ae.classes_)
                    proba_cls = reindex_proba(proba_cls, knn_cls.classes_)
                    y_test_late = y_all_alpha[idx_test_alpha]
                    for w_ae, label in [
                        (0.5, "LateFusion AE+CLS (avg)"),
                        (0.6, "LateFusion AE+CLS (0.6 AE)"),
                        (0.65, "LateFusion AE+CLS (0.65 AE)"),
                        (0.7, "LateFusion AE+CLS (0.7 AE)"),
                    ]:
                        w_cls = 1.0 - w_ae
                        combined_proba = w_ae * proba_ae + w_cls * proba_cls
                        y_pred_late = np.argmax(combined_proba, axis=1)
                        acc_lf = accuracy_score(y_test_late, y_pred_late)
                        b_acc_lf = balanced_accuracy_score(y_test_late, y_pred_late)
                        f1_lf = f1_score(y_test_late, y_pred_late, average='macro', zero_division=0)
                        print(f">>> KNN ({label}):")
                        print(f"    Acc: {acc_lf:.4f} | Balanced Acc: {b_acc_lf:.4f} | F1-macro: {f1_lf:.4f} | AUC: N/A")
                    # 三路晚融合：AE + CLS + PatchAvg（权重 0.5 / 0.3 / 0.2）；与上文双路相同，均使用 n_neighbors
                    if X_all_patch_alpha is not None:
                        try:
                            knn_patch = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=max_jobs)
                            knn_patch.fit(X_all_patch_alpha[idx_train_alpha], y_all_alpha[idx_train_alpha])
                            proba_patch = knn_patch.predict_proba(X_all_patch_alpha[idx_test_alpha])
                            proba_patch = reindex_proba(proba_patch, knn_patch.classes_)
                            combined_3 = 0.5 * proba_ae + 0.3 * proba_cls + 0.2 * proba_patch
                            y_pred_3 = np.argmax(combined_3, axis=1)
                            acc_3 = accuracy_score(y_test_late, y_pred_3)
                            b_acc_3 = balanced_accuracy_score(y_test_late, y_pred_3)
                            f1_3 = f1_score(y_test_late, y_pred_3, average='macro', zero_division=0)
                            print(f">>> KNN (LateFusion AE+CLS+Patch 0.5/0.3/0.2):")
                            print(f"    Acc: {acc_3:.4f} | Balanced Acc: {b_acc_3:.4f} | F1-macro: {f1_3:.4f} | AUC: N/A")
                        except Exception as e3:
                            print(f">>> [AlphaEarth] Three-way late fusion failed: {e3}")
                except Exception as e_lf:
                    print(f">>> [AlphaEarth] Late fusion failed: {e_lf}")

                # 3) AlphaEarth Avg + CLS + Patch-Avg（z-score 拼接）
                if X_all_patch_alpha is not None:
                    X_train_ae_cls_patch, X_test_ae_cls_patch = build_concat_features(
                        idx_train_alpha, idx_test_alpha,
                        [ae_avg, X_all_embed_alpha, X_all_patch_alpha],
                        name="AlphaEarth + CLS + PatchAvg"
                    )
                    if X_train_ae_cls_patch is not None:
                        self._eval_knn_metrics(
                            X_train_ae_cls_patch, y_all_alpha[idx_train_alpha],
                            X_test_ae_cls_patch, y_all_alpha[idx_test_alpha],
                            n_neighbors, "AlphaEarth+CLS+PatchAvg", average_method='macro'
                        )
                    # 3b) 三路 L2-norm 再拼接，避免 CLS/Patch 维数主导
                    X_train_ae_cls_patch_l2, X_test_ae_cls_patch_l2 = build_concat_l2_blocks(
                        idx_train_alpha, idx_test_alpha,
                        [ae_avg, X_all_embed_alpha, X_all_patch_alpha],
                        name="AlphaEarth + CLS + PatchAvg"
                    )
                    if X_train_ae_cls_patch_l2 is not None:
                        self._eval_knn_metrics(
                            X_train_ae_cls_patch_l2, y_all_alpha[idx_train_alpha],
                            X_test_ae_cls_patch_l2, y_all_alpha[idx_test_alpha],
                            n_neighbors, "AlphaEarth+CLS+PatchAvg (L2-norm)", average_method='macro'
                        )
                else:
                    print(">>> [AlphaEarth] Patch-avg features are not available from model.encode; "
                          "skip AlphaEarth+CLS+PatchAvg evaluation.")

            except Exception as e:
                print(f">>> [AlphaEarth Warning] AlphaEarth analysis failed: {e}")

        return {}

    def knn_probe_CropHarvest_Classification(self, model: torch.nn.Module, args: Any, n_neighbors: int = 5, imputator=None) -> Dict[str, float]:
        """
        CropHarvest 分类 KNN 探针（与其他分类任务保持同口径）。
        """
        print(f">>> [ExpProbe] Starting KNN CropHarvest Classification (k={n_neighbors})...")
        cls_min_samples = int(getattr(args, 'cls_min_samples_per_class', 10))
        cls_few_shot_k = getattr(args, 'cls_few_shot_k', None)
        cache_key = (
            f"CropHarvest_Classification_{args.data}_{args.seq_len}_{getattr(args, 'sampling_stride', 'default')}"
            f"_min{cls_min_samples}_few{cls_few_shot_k or 'ratio'}"
        )
        use_cached_baseline = cache_key in _baseline_cache

        _, data_loader = data_provider_CropHarvest_Classification(args, flag='CropHarvest_Classification', disable_ddp_split=True)
        backbone_model = self._get_backbone(model)
        backbone_model.eval()
        backbone_model.to(self.device)

        preds_embed_list, preds_patch_avg_list, labels_list = [], [], []
        if not use_cached_baseline:
            preds_raw_list = []

        with torch.no_grad():
            for batch in data_loader:
                batch_x = batch[0].float().to(self.device)
                batch_x_mark = batch[1].float().to(self.device)
                labels = batch[2].to(self.device)

                with autocast():
                    outputs = backbone_model.encode(batch_x, batch_x_mark, imputator=imputator)

                embedding = self._probe_cls_embedding(
                    outputs, args, seq_len=int(batch_x.shape[1])
                )
                preds_embed_list.append(embedding.detach().cpu().numpy())
                labels_list.append(labels.detach().cpu().numpy())

                patch_tokens = outputs.get('patch_tokens', None)
                if patch_tokens is not None:
                    if patch_tokens.ndim == 3:
                        patch_avg = patch_tokens.mean(dim=1)
                    elif patch_tokens.ndim == 4:
                        patch_avg = patch_tokens.mean(dim=1).mean(dim=1)
                    else:
                        patch_avg = None
                    if patch_avg is not None:
                        preds_patch_avg_list.append(patch_avg.detach().cpu().numpy())

                if not use_cached_baseline:
                    batch_x_filled = torch.nan_to_num(batch_x, nan=0.0)
                    raw_feat = batch_x_filled.mean(dim=1)
                    preds_raw_list.append(raw_feat.detach().cpu().numpy())

        X_all_embed = np.concatenate(preds_embed_list, axis=0)
        y_all = np.concatenate(labels_list, axis=0).reshape(-1)
        del preds_embed_list, labels_list

        X_all_patch_avg = None
        if len(preds_patch_avg_list) > 0:
            try:
                X_all_patch_avg = np.concatenate(preds_patch_avg_list, axis=0)
                print(f">>> [CropHarvest] Patch-avg features: X={X_all_patch_avg.shape}")
            except Exception as e:
                print(f">>> [CropHarvest Warning] Failed to concatenate patch-avg features: {e}")
                X_all_patch_avg = None
        preds_patch_avg_list = []

        y_all_before_class_filter = y_all.copy()
        idx_keep, y_remap = _filter_small_classes(y_all, min_samples_per_class=cls_min_samples)
        if len(idx_keep) == 0:
            print(f">>> [CropHarvest] No samples after filtering classes with < {cls_min_samples} samples. Skip.")
            return {}
        X_all_embed = X_all_embed[idx_keep]
        y_all = y_remap
        if X_all_patch_avg is not None:
            X_all_patch_avg = X_all_patch_avg[idx_keep]
        print(f">>> [CropHarvest] Embedding features: X={X_all_embed.shape}, y={y_all.shape}")

        if use_cached_baseline:
            cached_data = _baseline_cache[cache_key]
            X_all_raw = cached_data['X_all_raw']
            idx_train = cached_data['idx_train']
            idx_test = cached_data['idx_test']
            print(f">>> [CropHarvest Using cached baseline] Train: {len(idx_train)}, Test: {len(idx_test)}")
        else:
            X_all_raw = np.concatenate(preds_raw_list, axis=0)[idx_keep]
            cls_seed = int(getattr(args, 'cls_split_seed', 42))
            cls_ratio = float(getattr(args, 'cls_train_ratio', 0.8))
            if cls_few_shot_k is not None and cls_few_shot_k > 0:
                idx_train, idx_test = _get_few_shot_split(len(y_all), y_all, cls_few_shot_k, cls_seed)
                print(f">>> [CropHarvest Split] Few-shot k={cls_few_shot_k} per class, seed={cls_seed}")
            else:
                indices = np.arange(len(y_all))
                try:
                    idx_train, idx_test = train_test_split(
                        indices, train_size=cls_ratio, random_state=cls_seed, stratify=y_all,
                    )
                except ValueError:
                    idx_train, idx_test = train_test_split(indices, train_size=cls_ratio, random_state=cls_seed)
                print(f">>> [CropHarvest Split] train_ratio={cls_ratio}, seed={cls_seed}")

            _baseline_cache[cache_key] = {
                'X_all_raw': X_all_raw,
                'idx_train': idx_train,
                'idx_test': idx_test,
            }
            print(f">>> [CropHarvest Computing baseline] Train: {len(idx_train)}, Test: {len(idx_test)}")

        unique, counts = np.unique(y_all, return_counts=True)
        print(f">>> [CropHarvest] Class Distribution: {dict(zip(unique, counts))}")

        self._eval_knn_metrics(
            X_all_embed[idx_train],
            y_all[idx_train],
            X_all_embed[idx_test],
            y_all[idx_test],
            n_neighbors,
            "CropHarvest Encoded CLS",
            average_method='macro',
        )

        self._eval_knn_cls_plus_patch_avg(
            X_all_embed,
            X_all_patch_avg,
            idx_train,
            idx_test,
            y_all,
            n_neighbors,
            task_title_prefix="CropHarvest",
        )

        if use_cached_baseline:
            cached_result = _baseline_cache[cache_key].get('baseline_result_classification')
            if cached_result:
                print(cached_result, end='')
        else:
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            self._eval_knn_metrics(
                X_all_raw[idx_train],
                y_all[idx_train],
                X_all_raw[idx_test],
                y_all[idx_test],
                n_neighbors,
                "CropHarvest Raw Mean",
                average_method='macro',
            )
            baseline_result = buffer.getvalue()
            sys.stdout = old_stdout
            print(baseline_result, end='')
            _baseline_cache[cache_key]['baseline_result_classification'] = baseline_result

        # Optional AlphaEarth (GSE) quick comparison.
        use_alphaearth = getattr(args, 'use_alphaearth', False)
        if use_alphaearth:
            ae_path = getattr(args, 'alphaearth_path', None)
            if not ae_path:
                _ds_root = getattr(args, 'downstream_data_root', '/intelnvme01/ziyun/DownStreamTasks')
                ae_path = os.path.join(_ds_root, 'downstream_classification_task', 'cropharvest_gse_classification.npz')
            try:
                ae_npz = np.load(ae_path, allow_pickle=True)
                ae_data = _validate_alphaearth_array(ae_npz['data'], "CropHarvest AlphaEarth")
                ae_data = _alphaearth_data_after_class_filter(ae_data, idx_keep, "CropHarvest Classification + AlphaEarth")
                if ae_data.ndim == 3:
                    ae_avg = ae_data.mean(axis=1)
                elif ae_data.ndim == 2:
                    ae_avg = ae_data
                else:
                    raise ValueError(f"Unexpected CropHarvest AlphaEarth data ndim={ae_data.ndim}, expected 2 or 3.")
                self._eval_knn_metrics(
                    ae_avg[idx_train],
                    y_all[idx_train],
                    ae_avg[idx_test],
                    y_all[idx_test],
                    n_neighbors,
                    "CropHarvest AlphaEarthOnly",
                    average_method='macro',
                )
            except Exception as e:
                print(f">>> [AlphaEarth Warning] CropHarvest AlphaEarth analysis failed: {e}")

        return {}
    
    def knn_probe_GlanceTraining_Classification(self, model: torch.nn.Module, args: Any, n_neighbors: int = 5, imputator=None) -> Dict[str, float]:
        print(f">>> [ExpProbe] Starting KNN GlanceTraining Classification (k={n_neighbors})...")
        cloud_mask_ratio = float(getattr(args, "probe_cloud_mask_ratio", 0.0))
        cloud_mask_seed = int(getattr(args, "probe_cloud_mask_seed", 2026))
        if cloud_mask_ratio > 0:
            print(f">>> [CloudMask] ratio={cloud_mask_ratio} (seed={cloud_mask_seed})")
        cls_min_samples = int(getattr(args, 'cls_min_samples_per_class', 10))
        cls_few_shot_k = getattr(args, 'cls_few_shot_k', None)
        cache_key = (
            f"GlanceTraining_Classification_{args.data}_{args.seq_len}_{args.sampling_stride if hasattr(args, 'sampling_stride') else 'default'}"
            f"_min{cls_min_samples}_few{cls_few_shot_k or 'ratio'}"
        )
        use_cached_baseline = cache_key in _baseline_cache
        
        # 关键修复：KNN Probe只在rank 0运行，需要禁用DDP切分以获取全部数据
        _, data_loader = data_provider_GlanceTraining_Classification(args, flag='GlanceTraining_Classification', disable_ddp_split=True)
        print('args.num workers', args.num_workers)
        backbone_model = self._get_backbone(model)
        backbone_model.eval()
        backbone_model.to(self.device)

        preds_embed_list, labels_list, preds_patch_avg_list = [], [], []
        
        # 只在第一次或需要重新计算baseline时提取原始特征
        if not use_cached_baseline:
            preds_raw_list = []

        with torch.no_grad():
            for step, batch in enumerate(data_loader):
                batch_x = batch[0].float().to(self.device)
                batch_x_mark = batch[1].float().to(self.device)
                labels = batch[2].to(self.device)

                # simulate clouds BEFORE encode (and before raw baselines)
                batch_x = self._apply_probe_cloud_mask(batch_x, cloud_mask_ratio, cloud_mask_seed, step)

                with autocast():
                    outputs = backbone_model.encode(batch_x, batch_x_mark, imputator=imputator)

                # 1. Model Feature (CLS Token)
                embedding = self._probe_cls_embedding(
                    outputs, args, seq_len=int(batch_x.shape[1])
                )

                preds_embed_list.append(embedding.detach().cpu().numpy())
                labels_list.append(labels.detach().cpu().numpy())

                patch_tokens = outputs.get('patch_tokens', None)
                if patch_tokens is not None:
                    if patch_tokens.ndim == 3:
                        patch_avg = patch_tokens.mean(dim=1)
                    elif patch_tokens.ndim == 4:
                        patch_avg = patch_tokens.mean(dim=1).mean(dim=1)
                    else:
                        patch_avg = None
                    if patch_avg is not None:
                        preds_patch_avg_list.append(patch_avg.detach().cpu().numpy())
                
                # 2. Raw Feature (Mean over time) - 只在第一次计算
                if not use_cached_baseline:
                    batch_x_filled = torch.nan_to_num(batch_x, nan=0.0)
                    raw_feat = batch_x_filled.mean(dim=1)
                    preds_raw_list.append(raw_feat.detach().cpu().numpy())

        # ==========================================
        # 1. 处理 Embedding (模型特征) - 移除窗口合并逻辑
        # ==========================================
        X_all_embed = np.concatenate(preds_embed_list, axis=0) 
        y_all = np.concatenate(labels_list, axis=0).reshape(-1)
        del preds_embed_list, labels_list

        X_all_patch_avg = None
        if len(preds_patch_avg_list) > 0:
            try:
                X_all_patch_avg = np.concatenate(preds_patch_avg_list, axis=0)
                print(f">>> [GlanceTraining] Patch-avg features: X={X_all_patch_avg.shape}")
            except Exception as e:
                print(f">>> [GlanceTraining Warning] Failed to concatenate patch-avg features: {e}")
        preds_patch_avg_list = []

        print(f">>> Embedding features: X={X_all_embed.shape}, y={y_all.shape}")
        idx_keep, y_remap = _filter_small_classes(y_all, min_samples_per_class=cls_min_samples)
        if len(idx_keep) == 0:
            print(f">>> [GlanceTraining] No samples after filtering classes with < {cls_min_samples} samples. Skip.")
            return {}
        X_all_embed = X_all_embed[idx_keep]
        y_all = y_remap
        if X_all_patch_avg is not None:
            X_all_patch_avg = X_all_patch_avg[idx_keep]
        print(f">>> [GlanceTraining] After filtering small classes: N={len(y_all)}, n_class={len(np.unique(y_all))}")

        # ==========================================
        # 2. 处理 Raw Feature (原始特征)
        # ==========================================
        if use_cached_baseline:
            cached_data = _baseline_cache[cache_key]
            X_all_raw = cached_data['X_all_raw']
            idx_train = cached_data['idx_train']
            idx_test = cached_data['idx_test']
            print(f">>> [Using cached baseline] Train: {len(idx_train)}, Test: {len(idx_test)}")
        else:
            X_all_raw = np.concatenate(preds_raw_list, axis=0)
            del preds_raw_list
            X_all_raw = X_all_raw[idx_keep]
            cls_seed = int(getattr(args, 'cls_split_seed', 42))
            cls_ratio = float(getattr(args, 'cls_train_ratio', 0.8))
            if cls_few_shot_k is not None and cls_few_shot_k > 0:
                idx_train, idx_test = _get_few_shot_split(len(y_all), y_all, cls_few_shot_k, cls_seed)
                print(f">>> [Split] Few-shot k={cls_few_shot_k} per class, seed={cls_seed}")
            else:
                indices = np.arange(len(y_all))
                try:
                    idx_train, idx_test = train_test_split(indices, train_size=cls_ratio, random_state=cls_seed, stratify=y_all)
                except ValueError:
                    idx_train, idx_test = train_test_split(indices, train_size=cls_ratio, random_state=cls_seed)
                print(f">>> [Split] train_ratio={cls_ratio}, seed={cls_seed}")
            _baseline_cache[cache_key] = {'X_all_raw': X_all_raw, 'idx_train': idx_train, 'idx_test': idx_test}
            print(f">>> [Computing baseline] Train: {len(idx_train)}, Test: {len(idx_test)}")

        # Print Distribution
        unique, counts = np.unique(y_all, return_counts=True)
        print(f">>> Class Distribution: {dict(zip(unique, counts))}")

        # ==========================================
        # 3. 评估 (Eval)
        # ==========================================
        
        # 评估 Model Feature
        self._eval_knn_metrics(X_all_embed[idx_train], y_all[idx_train], 
                               X_all_embed[idx_test], y_all[idx_test], 
                               n_neighbors, "Encoded CLS", average_method='macro')

        self._eval_knn_cls_plus_patch_avg(
            X_all_embed,
            X_all_patch_avg,
            idx_train,
            idx_test,
            y_all,
            n_neighbors,
            task_title_prefix="",
        )

        # 评估 Raw Feature (Baseline)
        if use_cached_baseline:
            cached_result = _baseline_cache[cache_key].get('baseline_result_classification')
            if cached_result:
                print(cached_result, end='')
            else:
                # 理论上不该进这里，除非缓存结构不完整，但也补一个计算
                 pass 
        else:
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            # X_all_raw 形状: (N, C)，和 idx 对应
            self._eval_knn_metrics(X_all_raw[idx_train], y_all[idx_train], 
                                   X_all_raw[idx_test], y_all[idx_test], 
                                   n_neighbors, "Raw Mean", average_method='macro')
            baseline_result = buffer.getvalue()
            sys.stdout = old_stdout
            print(baseline_result, end='')
            _baseline_cache[cache_key]['baseline_result_classification'] = baseline_result

        # ==========================================
        # 4. 可选：AlphaEarth Embedding 分析（GlanceTraining）
        # ==========================================
        use_alphaearth = getattr(args, 'use_alphaearth', False)
        if use_alphaearth:
            print("\n>>> [AlphaEarth] Start AlphaEarth embeddings analysis for GlanceTraining Classification...")
            alphaearth_path = getattr(args, 'alphaearth_path', None)
            if not alphaearth_path:
                _ds_root = getattr(args, 'downstream_data_root', '/intelnvme01/ziyun/DownStreamTasks')
                alphaearth_path = _resolve_first_existing_path([
                    os.path.join(_ds_root, 'downstream_classification_task', 'glancetraining_gse_classification.npz'),
                    os.path.join(_ds_root, 'downstream_classification_task', 'glancetraining_alphaearth_classification.npz'),
                    os.path.join(_ds_root, 'GlanceTraining', 'ae_classification_dataset.npz'),
                ])
            if not alphaearth_path:
                print(
                    ">>> [AlphaEarth Warning] GlanceTraining AlphaEarth file not found. "
                    "Tried glancetraining_gse_classification.npz / "
                    "glancetraining_alphaearth_classification.npz / legacy ae_classification_dataset.npz."
                )
                return {}

            try:
                ae_npz = np.load(alphaearth_path, allow_pickle=True)
                ae_data = ae_npz['data']
                ae_time = ae_npz['time']    # (T_years,)
                # GlanceTraining AlphaEarth: data 形状为 (T_years, D, N_plots)，需要转成 (N, T_years, D)
                if ae_data.ndim == 3 and ae_data.shape[0] == ae_time.shape[0]:
                    # (T, D, N) -> (N, T, D)
                    ae_data = np.transpose(ae_data, (2, 0, 1))
                ae_data = np.nan_to_num(ae_data, nan=0.0, posinf=0.0, neginf=0.0)
                ae_labels = ae_npz.get('labels', ae_npz.get('class_ids', None))

                ae_data = _alphaearth_data_after_class_filter(
                    ae_data, idx_keep, "GlanceTraining Classification + AlphaEarth"
                )
                y_all_alpha = y_all
                X_all_embed_alpha = X_all_embed
                X_all_raw_alpha = X_all_raw
                idx_train_alpha = idx_train
                idx_test_alpha = idx_test
                if X_all_patch_avg is not None:
                    X_all_patch_alpha = X_all_patch_avg
                else:
                    X_all_patch_alpha = None

                # AlphaEarth 代表空间平均状态：如存在多年份/时间维，统一在时间维上做平均得到单一向量。
                if ae_data.ndim == 3:
                    ae_avg = ae_data.mean(axis=1).astype(np.float32)
                elif ae_data.ndim == 2:
                    ae_avg = ae_data.astype(np.float32)
                else:
                    raise ValueError(f"Unexpected GlanceTraining AlphaEarth data ndim={ae_data.ndim}, expected 2 or 3.")
                print(f">>> [AlphaEarth] GlanceTraining AlphaEarth ae_avg shape = {ae_avg.shape}")

                from sklearn.preprocessing import StandardScaler

                def _l2_norm_rows(X: np.ndarray) -> np.ndarray:
                    norm = np.linalg.norm(X, axis=1, keepdims=True)
                    norm = np.where(norm < 1e-8, 1.0, norm)
                    return X / norm

                def build_concat_features(train_idx, test_idx, feat_list, name: str):
                    train_blocks, test_blocks = [], []
                    for feat in feat_list:
                        if feat is None:
                            continue
                        scaler = StandardScaler()
                        feat_train = scaler.fit_transform(feat[train_idx])
                        feat_test = scaler.transform(feat[test_idx])
                        train_blocks.append(feat_train)
                        test_blocks.append(feat_test)
                    if not train_blocks:
                        return None, None
                    X_train_cat = np.concatenate(train_blocks, axis=1)
                    X_test_cat = np.concatenate(test_blocks, axis=1)
                    print(f">>> [AlphaEarth] {name} features: X_train={X_train_cat.shape}, X_test={X_test_cat.shape}")
                    return X_train_cat, X_test_cat

                def build_concat_l2_blocks(train_idx, test_idx, feat_list, name: str):
                    train_blocks, test_blocks = [], []
                    for feat in feat_list:
                        if feat is None:
                            continue
                        train_blocks.append(_l2_norm_rows(feat[train_idx]))
                        test_blocks.append(_l2_norm_rows(feat[test_idx]))
                    if not train_blocks:
                        return None, None
                    X_train_cat = np.concatenate(train_blocks, axis=1)
                    X_test_cat = np.concatenate(test_blocks, axis=1)
                    print(f">>> [AlphaEarth] {name} (L2-norm blocks) features: X_train={X_train_cat.shape}, X_test={X_test_cat.shape}")
                    return X_train_cat, X_test_cat

                print(">>> [AlphaEarth] GlanceTraining baseline Encoded CLS accuracy is reported above (without AlphaEarth).")

                # 1) 仅 AlphaEarth Avg
                X_train_ae_only, X_test_ae_only = build_concat_features(
                    idx_train_alpha, idx_test_alpha,
                    [ae_avg],
                    name="GlanceTraining AlphaEarth Only"
                )
                if X_train_ae_only is not None:
                    self._eval_knn_metrics(
                        X_train_ae_only, y_all_alpha[idx_train_alpha],
                        X_test_ae_only, y_all_alpha[idx_test_alpha],
                        n_neighbors, "GlanceTraining AlphaEarthOnly", average_method='macro'
                    )
                    self._eval_knn_metrics(
                        X_train_ae_only, y_all_alpha[idx_train_alpha],
                        X_test_ae_only, y_all_alpha[idx_test_alpha],
                        n_neighbors, "GlanceTraining AlphaEarthOnly (weights=distance)", average_method='macro', knn_weights='distance'
                    )

                # 2) AlphaEarth Avg + CLS（z-score 拼接）
                X_train_ae_cls, X_test_ae_cls = build_concat_features(
                    idx_train_alpha, idx_test_alpha,
                    [ae_avg, X_all_embed_alpha],
                    name="GlanceTraining AlphaEarth + CLS"
                )
                if X_train_ae_cls is not None:
                    self._eval_knn_metrics(
                        X_train_ae_cls, y_all_alpha[idx_train_alpha],
                        X_test_ae_cls, y_all_alpha[idx_test_alpha],
                        n_neighbors, "GlanceTraining AlphaEarth+CLS", average_method='macro'
                    )

                # 2b) AlphaEarth + CLS（按块 L2 归一化再拼接）
                X_train_ae_cls_l2, X_test_ae_cls_l2 = build_concat_l2_blocks(
                    idx_train_alpha, idx_test_alpha,
                    [ae_avg, X_all_embed_alpha],
                    name="GlanceTraining AlphaEarth + CLS"
                )
                if X_train_ae_cls_l2 is not None:
                    self._eval_knn_metrics(
                        X_train_ae_cls_l2, y_all_alpha[idx_train_alpha],
                        X_test_ae_cls_l2, y_all_alpha[idx_test_alpha],
                        n_neighbors, "GlanceTraining AlphaEarth+CLS (L2-norm)", average_method='macro'
                    )
                    self._eval_knn_metrics(
                        X_train_ae_cls_l2, y_all_alpha[idx_train_alpha],
                        X_test_ae_cls_l2, y_all_alpha[idx_test_alpha],
                        n_neighbors, "GlanceTraining AlphaEarth+CLS (L2-norm, weights=distance)", average_method='macro', knn_weights='distance'
                    )

                # 2c) 晚融合（与 LCMAP Classification 对齐，便于跨数据集对比）
                try:
                    max_jobs = min(8, os.cpu_count() or 4)
                    knn_ae = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=max_jobs)
                    knn_cls = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=max_jobs)
                    knn_ae.fit(X_train_ae_only, y_all_alpha[idx_train_alpha])
                    knn_cls.fit(X_all_embed_alpha[idx_train_alpha], y_all_alpha[idx_train_alpha])
                    proba_ae = knn_ae.predict_proba(X_test_ae_only)
                    proba_cls = knn_cls.predict_proba(X_all_embed_alpha[idx_test_alpha])
                    unified_classes = np.unique(y_all_alpha)

                    def reindex_proba(proba, classes):
                        cols = []
                        for c in unified_classes:
                            idx = np.where(classes == c)[0]
                            cols.append(proba[:, idx[0]] if len(idx) > 0 else np.zeros(proba.shape[0]))
                        return np.stack(cols, axis=1)

                    proba_ae = reindex_proba(proba_ae, knn_ae.classes_)
                    proba_cls = reindex_proba(proba_cls, knn_cls.classes_)
                    y_test_late = y_all_alpha[idx_test_alpha]
                    for w_ae, label in [
                        (0.5, "GlanceTraining LateFusion AE+CLS (avg)"),
                        (0.6, "GlanceTraining LateFusion AE+CLS (0.6 AE)"),
                        (0.65, "GlanceTraining LateFusion AE+CLS (0.65 AE)"),
                        (0.7, "GlanceTraining LateFusion AE+CLS (0.7 AE)"),
                    ]:
                        w_cls = 1.0 - w_ae
                        combined_proba = w_ae * proba_ae + w_cls * proba_cls
                        y_pred_late = np.argmax(combined_proba, axis=1)
                        print(f">>> KNN ({label}):")
                        print(
                            f"    Acc: {accuracy_score(y_test_late, y_pred_late):.4f} | "
                            f"Balanced Acc: {balanced_accuracy_score(y_test_late, y_pred_late):.4f} | "
                            f"F1-macro: {f1_score(y_test_late, y_pred_late, average='macro', zero_division=0):.4f} | AUC: N/A"
                        )
                    if X_all_patch_alpha is not None:
                        try:
                            knn_patch = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=max_jobs)
                            knn_patch.fit(X_all_patch_alpha[idx_train_alpha], y_all_alpha[idx_train_alpha])
                            proba_patch = knn_patch.predict_proba(X_all_patch_alpha[idx_test_alpha])
                            proba_patch = reindex_proba(proba_patch, knn_patch.classes_)
                            combined_3 = 0.5 * proba_ae + 0.3 * proba_cls + 0.2 * proba_patch
                            y_pred_3 = np.argmax(combined_3, axis=1)
                            print(f">>> KNN (GlanceTraining LateFusion AE+CLS+Patch 0.5/0.3/0.2):")
                            print(
                                f"    Acc: {accuracy_score(y_test_late, y_pred_3):.4f} | "
                                f"Balanced Acc: {balanced_accuracy_score(y_test_late, y_pred_3):.4f} | "
                                f"F1-macro: {f1_score(y_test_late, y_pred_3, average='macro', zero_division=0):.4f} | AUC: N/A"
                            )
                        except Exception as e3:
                            print(f">>> [AlphaEarth] GlanceTraining three-way late fusion failed: {e3}")
                except Exception as e_lf:
                    print(f">>> [AlphaEarth] GlanceTraining late fusion failed: {e_lf}")

                # 3) AlphaEarth + CLS + Patch-Avg（z-score / L2 拼接）
                if X_all_patch_alpha is not None:
                    X_train_ae_cls_patch, X_test_ae_cls_patch = build_concat_features(
                        idx_train_alpha, idx_test_alpha,
                        [ae_avg, X_all_embed_alpha, X_all_patch_alpha],
                        name="GlanceTraining AlphaEarth + CLS + PatchAvg",
                    )
                    if X_train_ae_cls_patch is not None:
                        self._eval_knn_metrics(
                            X_train_ae_cls_patch, y_all_alpha[idx_train_alpha],
                            X_test_ae_cls_patch, y_all_alpha[idx_test_alpha],
                            n_neighbors, "GlanceTraining AlphaEarth+CLS+PatchAvg", average_method='macro'
                        )
                    X_train_ae_cls_patch_l2, X_test_ae_cls_patch_l2 = build_concat_l2_blocks(
                        idx_train_alpha, idx_test_alpha,
                        [ae_avg, X_all_embed_alpha, X_all_patch_alpha],
                        name="GlanceTraining AlphaEarth + CLS + PatchAvg",
                    )
                    if X_train_ae_cls_patch_l2 is not None:
                        self._eval_knn_metrics(
                            X_train_ae_cls_patch_l2, y_all_alpha[idx_train_alpha],
                            X_test_ae_cls_patch_l2, y_all_alpha[idx_test_alpha],
                            n_neighbors, "GlanceTraining AlphaEarth+CLS+PatchAvg (L2-norm)", average_method='macro'
                        )
                else:
                    print(
                        ">>> [AlphaEarth] GlanceTraining: patch-avg not available from encode; "
                        "skip AlphaEarth+CLS+PatchAvg concat and three-way late fusion (AE+CLS late fusion above still runs)."
                    )
            except Exception as e:
                print(f">>> [AlphaEarth Warning] GlanceTraining AlphaEarth analysis failed: {e}")

        return {}

    def knn_probe_LCMAP_Segmentation(self, model: torch.nn.Module, args: Any, n_neighbors: int = 5) -> Dict[str, float]:
        """LCMAP 探针：与 GlobalTree 对齐，增加分段相关任务（变化次数、变化类型），便于对比序列分段能力。"""
        print(f">>> [ExpProbe] Starting KNN LCMAP Segmentation (k={n_neighbors})...")
        print(f">>> [LCMAP_Segmentation] Focus: sequence segmentation (breakpoints) + yearly classification.")

        # 生成缓存键
        cache_key = f"LCMAP_Segmentation_year_seg_cls_{args.data}_{args.seq_len}_{args.sampling_stride if hasattr(args, 'sampling_stride') else 'default'}"
        use_cached_baseline = cache_key in _baseline_cache

        # 关键修复：KNN Probe只在rank 0运行，需要禁用DDP切分以获取全部数据
        _, data_loader = data_provider_LCMAP_Segmentation(args, flag='LCMAP_Segmentation', disable_ddp_split=True)
        backbone_model = self._get_backbone(model)
        backbone_model.eval()
        backbone_model.to(self.device)

        # Lists for aggregation
        cls_embedding_list = []
        patch_embedding_list, yearly_labels_list = [], [] 
        
        # 只在第一次或需要重新计算baseline时提取原始特征
        if not use_cached_baseline:
            raw_feat_list, raw_patch_list = [], [] # Baselines

        with torch.no_grad():
            for batch in data_loader:
                batch_x = batch[0].float().to(self.device) # [B, T, C]
                batch_x_mark = batch[1].float().to(self.device)
                labels = batch[2].to(self.device) # [B, Years]

                with autocast():
                    outputs = backbone_model.encode(batch_x, batch_x_mark)

                cls_embedding = self._probe_cls_embedding(
                    outputs, args, seq_len=int(batch_x.shape[1])
                )
                patch_embedding = outputs['patch_tokens'] # [B, T, D]

                # --- 按年切段再池化：时间轴等分成 num_years 段，每段内 mean，与「年」对齐 ---
                num_years_target = labels.shape[1]
                patch_embedding_agg = self._segment_pool_by_years(patch_embedding, num_years_target)  # [B, num_years, D]
                # 每年表征拼接 CLS：整序列全局状态，利于逐年 KNN
                cls_broadcast = cls_embedding.unsqueeze(1).expand(-1, num_years_target, -1)  # [B, num_years, D]
                patch_embedding_agg = torch.cat([patch_embedding_agg, cls_broadcast], dim=-1)  # [B, num_years, 2*D]

                # --- Collect ---
                cls_embedding_list.append(cls_embedding.detach().cpu().numpy())
                patch_embedding_list.append(patch_embedding_agg.detach().cpu().numpy())
                yearly_labels_list.append(labels.detach().cpu().numpy())
                
                # B. Raw Features Baseline（同样按年切段池化，每年拼上整序列均值作为“全局状态”）
                if not use_cached_baseline:
                    batch_x_filled = torch.nan_to_num(batch_x, nan=0.0)
                    raw_feat_global = batch_x_filled.mean(dim=1)  # [B, C]
                    raw_patch_agg = self._segment_pool_by_years(batch_x_filled, num_years_target)  # [B, num_years, C]
                    raw_global_broadcast = raw_feat_global.unsqueeze(1).expand(-1, num_years_target, -1)
                    raw_patch_agg = torch.cat([raw_patch_agg, raw_global_broadcast], dim=-1)  # [B, num_years, 2*C]
                    raw_feat_list.append(raw_feat_global.detach().cpu().numpy())
                    raw_patch_list.append(raw_patch_agg.detach().cpu().numpy())

        # === 1. Concatenate Batch ===
        X_cls = np.concatenate(cls_embedding_list, axis=0)           # (N, D)
        X_patch = np.concatenate(patch_embedding_list, axis=0)       # (N, T_win, D)
        y_yearly = np.concatenate(yearly_labels_list, axis=0)        # (N, T_win)

        del cls_embedding_list, patch_embedding_list, yearly_labels_list

        # ==========================================
        # === 2. 移除窗口合并逻辑：每个序列都是独立样本 ===
        # ==========================================
        print(f">>> Features: X_cls={X_cls.shape}, X_patch={X_patch.shape}, y_yearly={y_yearly.shape}")

        # 变化/分段标签（与 GlobalTree 对齐：相邻年不同算一次变化）
        y_change = (y_yearly != 0).any(axis=1).astype(np.int64)  # LCMAP 原有：任一年非零即变化
        if y_yearly.shape[1] > 1:
            diff = (y_yearly[:, 1:] != y_yearly[:, :-1])
            num_transitions = diff.sum(axis=1).astype(np.int64)
            max_trans = min(5, y_yearly.shape[1] - 1)  # 6 年最多 5 次变化
            y_num_transitions = np.minimum(num_transitions, max_trans).astype(np.int64)
            first_last_same = (y_yearly[:, 0] == y_yearly[:, -1])
            aba_mask = (num_transitions == 2) & first_last_same
            y_change_type = np.where(num_transitions == 0, 0,
                           np.where(num_transitions == 1, 1,
                           np.where(aba_mask, 2, 3))).astype(np.int64)
        else:
            y_num_transitions = np.zeros(y_yearly.shape[0], dtype=np.int64)
            y_change_type = np.zeros(y_yearly.shape[0], dtype=np.int64)

        print(f">>> Segmentation labels: y_change={y_change.shape}, y_num_transitions={y_num_transitions.shape}, y_change_type={y_change_type.shape}")
        unique, counts = np.unique(y_num_transitions, return_counts=True)
        print(f">>> [LCMAP] Num transitions distribution: {dict(zip(unique, counts))} (0/1/2/.../5+)")
        unique, counts = np.unique(y_change_type, return_counts=True)
        print(f">>> [LCMAP] Change type distribution: {dict(zip(unique, counts))} (0=stable, 1=one, 2=ABA, 3=multi)")

        # ==========================================
        # === 3. Baseline Data (移除窗口合并逻辑) ===
        # ==========================================
        if use_cached_baseline:
            cached_data = _baseline_cache[cache_key]
            X_raw = cached_data['X_raw']
            X_raw_patch = cached_data['X_raw_patch']
            idx_train = cached_data['idx_train']
            idx_test = cached_data['idx_test']
            print(f">>> [Using cached baseline] Train: {len(idx_train)}")
        else:
            # 处理 Raw Features
            X_raw = np.concatenate(raw_feat_list, axis=0) # (N, C)
            X_raw_patch = np.concatenate(raw_patch_list, axis=0) # (N, T, C)
            del raw_feat_list, raw_patch_list

            # 移除窗口合并逻辑：每个序列都是独立样本
            # X_raw 形状: (N, C)，X_raw_patch 形状: (N, T, C)

            # Split (基于 y_change)，统一采用 80% 训练 / 20% 测试
            indices = np.arange(len(y_change))
            try:
                idx_train, idx_test = train_test_split(
                    indices,
                    train_size=0.8,
                    random_state=42,
                    stratify=y_change
                )
            except ValueError:
                idx_train, idx_test = train_test_split(
                    indices,
                    train_size=0.8,
                    random_state=42
                )
            
            # Cache
            _baseline_cache[cache_key] = {
                'X_raw': X_raw,
                'X_raw_patch': X_raw_patch,
                'idx_train': idx_train,
                'idx_test': idx_test
            }
            print(f">>> [Computing baseline] Train: {len(idx_train)}")

        # ========== 分段任务（与 GlobalTree 对齐） ==========
        print(f"\n>>> [LCMAP_Segmentation] === Segmentation: breakpoints / change structure ===")

        # Task 1: Change Detection (Binary)
        print(f"\n>>> [LCMAP_Segmentation] Task 1: Change Detection (Binary)")
        self._eval_knn_metrics(X_cls[idx_train], y_change[idx_train], 
                               X_cls[idx_test], y_change[idx_test], 
                               n_neighbors, "CLS Token", average_method='binary')
        if use_cached_baseline:
             cached_res = cached_data.get('baseline_result_segmentation_task1')
             if cached_res: print(cached_res, end='')
        else:
            old_stdout = sys.stdout
            try:
                sys.stdout = buffer = io.StringIO()
                self._eval_knn_metrics(X_raw[idx_train], y_change[idx_train], 
                                       X_raw[idx_test], y_change[idx_test], 
                                       n_neighbors, "Raw Mean", average_method='binary')
                res = buffer.getvalue()
                _baseline_cache[cache_key]['baseline_result_segmentation_task1'] = res
            finally:
                sys.stdout = old_stdout
            print(res, end='')

        # Task 2: Num Transitions (0/1/2/.../5+) —— 变化次数，评估分段数量
        if len(np.unique(y_num_transitions)) >= 2:
            print(f"\n>>> [LCMAP_Segmentation] Task 2: Num Transitions (0/1/2/.../5+) [segment count - 1]")
            self._eval_knn_metrics(
                X_cls[idx_train], y_num_transitions[idx_train],
                X_cls[idx_test], y_num_transitions[idx_test],
                n_neighbors, "LCMAP Num Transitions CLS", average_method='macro',
            )
            self._eval_knn_metrics(
                X_raw[idx_train], y_num_transitions[idx_train],
                X_raw[idx_test], y_num_transitions[idx_test],
                n_neighbors, "LCMAP Num Transitions Raw Mean", average_method='macro',
            )
        else:
            print(f"\n>>> [LCMAP_Segmentation] Task 2: Num Transitions skipped (single class)")

        # Task 3: Change Type (Stable / One / ABA / Multi)
        if len(np.unique(y_change_type)) >= 2:
            print(f"\n>>> [LCMAP_Segmentation] Task 3: Change Type (Stable / One / ABA / Multi)")
            self._eval_knn_metrics(
                X_cls[idx_train], y_change_type[idx_train],
                X_cls[idx_test], y_change_type[idx_test],
                n_neighbors, "LCMAP Change Type CLS", average_method='macro',
            )
            self._eval_knn_metrics(
                X_raw[idx_train], y_change_type[idx_train],
                X_raw[idx_test], y_change_type[idx_test],
                n_neighbors, "LCMAP Change Type Raw Mean", average_method='macro',
            )
        else:
            print(f"\n>>> [LCMAP_Segmentation] Task 3: Change Type skipped (single class)")

        # ========== 补充：逐年分类 ==========
        print(f"\n>>> [LCMAP_Segmentation] === Supplementary: Yearly Classification (Plot-wise Split) ===")
        
        # Flatten Helper: [N, Years, D] -> [N*Years, D]
        def flatten_data(X_3d, y_2d, idx):
            X_sub = X_3d[idx]
            y_sub = y_2d[idx]
            return X_sub.reshape(-1, X_sub.shape[-1]), y_sub.reshape(-1)

        X_train_p, y_train_p = flatten_data(X_patch, y_yearly, idx_train)
        X_test_p, y_test_p   = flatten_data(X_patch, y_yearly, idx_test)
        
        X_train_rp, _        = flatten_data(X_raw_patch, y_yearly, idx_train)
        X_test_rp, _         = flatten_data(X_raw_patch, y_yearly, idx_test)

        self._eval_knn_metrics(X_train_p, y_train_p, X_test_p, y_test_p, 
                               n_neighbors, "Patch Tok", average_method='macro')

        # PCA 降维后再 KNN（缓解高维对 KNN 的不利）
        for X_tr, X_te, y_tr, y_te, name in [
            (X_train_p, X_test_p, y_train_p, y_test_p, "Patch Tok (PCA 0.95)"),
            (X_train_rp, X_test_rp, y_train_p, y_test_p, "Raw Input (PCA 0.95)"),
        ]:
            pca = PCA(n_components=0.95, random_state=42)
            pca.fit(X_tr)
            X_tr_pca = pca.transform(X_tr)
            X_te_pca = pca.transform(X_te)
            print(f">>> [PCA] {name}: dim {X_tr.shape[1]} -> {X_tr_pca.shape[1]}, var_ratio={pca.explained_variance_ratio_.sum():.4f}")
            self._eval_knn_metrics(X_tr_pca, y_tr, X_te_pca, y_te, n_neighbors, name, average_method='macro')
        
        # Baseline Task 2（原始 Raw Input 结果，用于缓存）
        if use_cached_baseline:
             cached_res = cached_data.get('baseline_result_segmentation_task2')
             if cached_res: print(cached_res, end='')
        else:
            old_stdout = sys.stdout
            try:
                sys.stdout = buffer = io.StringIO()
                self._eval_knn_metrics(X_train_rp, y_train_p, X_test_rp, y_test_p, 
                                       n_neighbors, "Raw Input", average_method='macro')
                res = buffer.getvalue()
                _baseline_cache[cache_key]['baseline_result_segmentation_task2'] = res
            finally:
                sys.stdout = old_stdout
            print(res, end='')

        return {}

    def knn_probe_GlobalTree_Segmentation(self, model: torch.nn.Module, args: Any, n_neighbors: int = 5) -> Dict[str, float]:
        """
        GlobalTree 探针：重点关注模型能否正确将序列分段（breakpoints / 变化结构），
        而非分段后的逐年分类。分段任务：是否有变化、变化次数、变化类型(含ABA)；逐年分类仅作补充。
        """
        print(f">>> [ExpProbe] Starting KNN GlobalTree Segmentation (k={n_neighbors})...")
        print(f">>> [GlobalTree_Segmentation] Focus: sequence segmentation (breakpoints), not segment classification.")

        cache_key = f"GlobalTree_Segmentation_year_seg_cls_{args.data}_{args.seq_len}_{getattr(args, 'sampling_stride', 'default')}"
        use_cached_baseline = cache_key in _baseline_cache

        _, data_loader = data_provider_GlobalTree_Segmentation(
            args, flag="GlobalTree_Segmentation", disable_ddp_split=True
        )
        backbone_model = self._get_backbone(model)
        backbone_model.eval()
        backbone_model.to(self.device)

        cls_embedding_list: List[np.ndarray] = []
        patch_embedding_list: List[np.ndarray] = []
        yearly_labels_list: List[np.ndarray] = []

        if not use_cached_baseline:
            raw_feat_list: List[np.ndarray] = []
            raw_patch_list: List[np.ndarray] = []

        with torch.no_grad():
            for batch in data_loader:
                batch_x = batch[0].float().to(self.device)
                batch_x_mark = batch[1].float().to(self.device)
                labels = batch[2].to(self.device)  # [B, Years]

                with autocast():
                    outputs = backbone_model.encode(batch_x, batch_x_mark)

                cls_embedding = self._probe_cls_embedding(
                    outputs, args, seq_len=int(batch_x.shape[1])
                )
                patch_embedding = outputs["patch_tokens"]

                num_years_target = labels.shape[1]
                # 按年切段再池化：与 LCMAP 一致，每年对应时间轴上一段，段内 mean
                patch_embedding_agg = self._segment_pool_by_years(patch_embedding, num_years_target)  # [B, num_years, D]
                # 每年表征拼接 CLS（整序列基本状态）
                cls_broadcast = cls_embedding.unsqueeze(1).expand(-1, num_years_target, -1)
                patch_embedding_agg = torch.cat([patch_embedding_agg, cls_broadcast], dim=-1)  # [B, num_years, 2*D]

                cls_embedding_list.append(cls_embedding.detach().cpu().numpy())
                patch_embedding_list.append(patch_embedding_agg.detach().cpu().numpy())
                yearly_labels_list.append(labels.detach().cpu().numpy())

                if not use_cached_baseline:
                    batch_x_filled = torch.nan_to_num(batch_x, nan=0.0)
                    raw_feat_global = batch_x_filled.mean(dim=1)
                    raw_patch_agg = self._segment_pool_by_years(batch_x_filled, num_years_target)
                    raw_global_broadcast = raw_feat_global.unsqueeze(1).expand(-1, num_years_target, -1)
                    raw_patch_agg = torch.cat([raw_patch_agg, raw_global_broadcast], dim=-1)  # [B, num_years, 2*C]
                    raw_feat_list.append(raw_feat_global.detach().cpu().numpy())
                    raw_patch_list.append(raw_patch_agg.detach().cpu().numpy())

        # 若数据加载器没有产生任何 batch，直接跳过本次 Probe，避免 concat 报错
        if len(cls_embedding_list) == 0:
            print(">>> [GlobalTree_Segmentation] No batches found, skip KNN probe for this task.")
            return {}

        X_cls = np.concatenate(cls_embedding_list, axis=0)
        X_patch = np.concatenate(patch_embedding_list, axis=0)
        y_yearly = np.concatenate(yearly_labels_list, axis=0)

        del cls_embedding_list, patch_embedding_list, yearly_labels_list

        print(f">>> [GlobalTree_Segmentation] Features: X_cls={X_cls.shape}, X_patch={X_patch.shape}, y_yearly={y_yearly.shape}")

        # 分段相关标签（关注：能否正确分段，不关注分段后类别）
        if y_yearly.shape[1] > 1:
            diff = (y_yearly[:, 1:] != y_yearly[:, :-1])  # [N, Years-1]
            num_transitions = diff.sum(axis=1).astype(np.int64)  # 变化次数 = 分段数 - 1
            y_change = (num_transitions > 0).astype(np.int64)
            # 纯分段数量：0/1/2/3+ 次变化（用于评估“分段对不对”）
            y_num_transitions = np.minimum(num_transitions, 3).astype(np.int64)  # 0,1,2,3+
            # 变化类型（含结构）：0=稳定, 1=一次变化, 2=ABA, 3=其它多次变化
            first_last_same = (y_yearly[:, 0] == y_yearly[:, -1])
            aba_mask = (num_transitions == 2) & first_last_same
            y_change_type = np.where(num_transitions == 0, 0,
                           np.where(num_transitions == 1, 1,
                           np.where(aba_mask, 2, 3))).astype(np.int64)
        else:
            y_change = np.zeros(y_yearly.shape[0], dtype=np.int64)
            y_num_transitions = np.zeros(y_yearly.shape[0], dtype=np.int64)
            y_change_type = np.zeros(y_yearly.shape[0], dtype=np.int64)

        print(f">>> [GlobalTree_Segmentation] Segmentation labels: y_change={y_change.shape}, y_num_transitions={y_num_transitions.shape}, y_change_type={y_change_type.shape}")
        unique, counts = np.unique(y_num_transitions, return_counts=True)
        print(f">>> [GlobalTree_Segmentation] Num transitions distribution: {dict(zip(unique, counts))} (0/1/2/3+)")
        unique, counts = np.unique(y_change_type, return_counts=True)
        print(f">>> [GlobalTree_Segmentation] Change type distribution: {dict(zip(unique, counts))} (0=stable, 1=one_change, 2=ABA, 3=multi_change)")

        if use_cached_baseline:
            cached_data = _baseline_cache[cache_key]
            X_raw = cached_data["X_raw"]
            X_raw_patch = cached_data["X_raw_patch"]
            idx_train = cached_data["idx_train"]
            idx_test = cached_data["idx_test"]
            print(f">>> [GlobalTree_Segmentation Using cached baseline] Train: {len(idx_train)}")
        else:
            X_raw = np.concatenate(raw_feat_list, axis=0)
            X_raw_patch = np.concatenate(raw_patch_list, axis=0)
            del raw_feat_list, raw_patch_list

            indices = np.arange(len(y_change))
            try:
                idx_train, idx_test = train_test_split(
                    indices,
                    train_size=0.8,
                    random_state=42,
                    stratify=y_change,
                )
            except ValueError:
                idx_train, idx_test = train_test_split(
                    indices,
                    train_size=0.8,
                    random_state=42,
                )

            _baseline_cache[cache_key] = {
                "X_raw": X_raw,
                "X_raw_patch": X_raw_patch,
                "idx_train": idx_train,
                "idx_test": idx_test,
            }
            print(f">>> [GlobalTree_Segmentation Computing baseline] Train: {len(idx_train)}")

        # ========== 重点：序列分段（能否正确分出段数/变化次数） ==========
        print("\n>>> [GlobalTree_Segmentation] === Segmentation (main): breakpoints / change structure ===")

        # Task 1: 是否有变化 (二分类)
        print("\n>>> [GlobalTree_Segmentation] Task 1: Change Detection (Binary)")
        self._eval_knn_metrics(
            X_cls[idx_train],
            y_change[idx_train],
            X_cls[idx_test],
            y_change[idx_test],
            n_neighbors,
            "GT Segmentation CLS",
            average_method="binary",
        )
        if use_cached_baseline:
            cached_res = cached_data.get("baseline_result_gt_segmentation_task1")
            if cached_res:
                print(cached_res, end="")
        else:
            old_stdout = sys.stdout
            try:
                sys.stdout = buffer = io.StringIO()
                self._eval_knn_metrics(
                    X_raw[idx_train],
                    y_change[idx_train],
                    X_raw[idx_test],
                    y_change[idx_test],
                    n_neighbors,
                    "GT Segmentation Raw Mean",
                    average_method="binary",
                )
                res = buffer.getvalue()
                _baseline_cache[cache_key]["baseline_result_gt_segmentation_task1"] = res
            finally:
                sys.stdout = old_stdout
            print(res, end="")

        # Task 2: 变化次数 (0/1/2/3+) —— 直接评估“分段数量”是否预测对
        if len(np.unique(y_num_transitions)) >= 2:
            print("\n>>> [GlobalTree_Segmentation] Task 2: Num Transitions (0/1/2/3+) [segment count - 1]")
            self._eval_knn_metrics(
                X_cls[idx_train],
                y_num_transitions[idx_train],
                X_cls[idx_test],
                y_num_transitions[idx_test],
                n_neighbors,
                "GT Num Transitions CLS",
                average_method="macro",
            )
            self._eval_knn_metrics(
                X_raw[idx_train],
                y_num_transitions[idx_train],
                X_raw[idx_test],
                y_num_transitions[idx_test],
                n_neighbors,
                "GT Num Transitions Raw Mean",
                average_method="macro",
            )
        else:
            print("\n>>> [GlobalTree_Segmentation] Task 2: Num Transitions skipped (single class)")

        # Task 3: 变化类型（稳定 / 一次 / ABA / 其它多次）
        if len(np.unique(y_change_type)) >= 2:
            print("\n>>> [GlobalTree_Segmentation] Task 3: Change Type (Stable / One / ABA / Multi)")
            self._eval_knn_metrics(
                X_cls[idx_train],
                y_change_type[idx_train],
                X_cls[idx_test],
                y_change_type[idx_test],
                n_neighbors,
                "GT Change Type CLS",
                average_method="macro",
            )
            self._eval_knn_metrics(
                X_raw[idx_train],
                y_change_type[idx_train],
                X_raw[idx_test],
                y_change_type[idx_test],
                n_neighbors,
                "GT Change Type Raw Mean",
                average_method="macro",
            )
        else:
            print("\n>>> [GlobalTree_Segmentation] Task 3: Change Type skipped (single class)")

        # ========== 补充：分段后逐年分类（已做过，此处仅作参考） ==========
        print("\n>>> [GlobalTree_Segmentation] === Supplementary: Yearly classification (segment labels) ===")

        def flatten_data_globaltree(X_3d: np.ndarray, y_2d: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            X_sub = X_3d[idx]
            y_sub = y_2d[idx]
            return X_sub.reshape(-1, X_sub.shape[-1]), y_sub.reshape(-1)

        X_train_p, y_train_p = flatten_data_globaltree(X_patch, y_yearly, idx_train)
        X_test_p, y_test_p = flatten_data_globaltree(X_patch, y_yearly, idx_test)

        X_train_rp, _ = flatten_data_globaltree(X_raw_patch, y_yearly, idx_train)
        X_test_rp, _ = flatten_data_globaltree(X_raw_patch, y_yearly, idx_test)

        self._eval_knn_metrics(
            X_train_p,
            y_train_p,
            X_test_p,
            y_test_p,
            n_neighbors,
            "GT Segmentation Patch Tok",
            average_method="macro",
        )

        # PCA 降维后再 KNN
        for X_tr, X_te, y_tr, y_te, name in [
            (X_train_p, X_test_p, y_train_p, y_test_p, "GT Segmentation Patch Tok (PCA 0.95)"),
            (X_train_rp, X_test_rp, y_train_p, y_test_p, "GT Segmentation Raw Input (PCA 0.95)"),
        ]:
            pca = PCA(n_components=0.95, random_state=42)
            pca.fit(X_tr)
            X_tr_pca = pca.transform(X_tr)
            X_te_pca = pca.transform(X_te)
            print(f">>> [PCA] {name}: dim {X_tr.shape[1]} -> {X_tr_pca.shape[1]}, var_ratio={pca.explained_variance_ratio_.sum():.4f}")
            self._eval_knn_metrics(X_tr_pca, y_tr, X_te_pca, y_te, n_neighbors, name, average_method="macro")

        if use_cached_baseline:
            cached_res = cached_data.get("baseline_result_gt_segmentation_task2")
            if cached_res:
                print(cached_res, end="")
        else:
            old_stdout = sys.stdout
            try:
                sys.stdout = buffer = io.StringIO()
                self._eval_knn_metrics(
                    X_train_rp,
                    y_train_p,
                    X_test_rp,
                    y_test_p,
                    n_neighbors,
                    "GT Segmentation Raw Input",
                    average_method="macro",
                )
                res = buffer.getvalue()
                _baseline_cache[cache_key]["baseline_result_gt_segmentation_task2"] = res
            finally:
                sys.stdout = old_stdout
            print(res, end="")

        return {}

    def knn_probe_GlanceTraining_Segmentation(self, model: torch.nn.Module, args: Any, n_neighbors: int = 5, imputator=None) -> Dict[str, float]:
        """
        GlanceTraining 二元变化检测（序列级）KNN 探针.

        使用:
        - CLS Token 特征做 KNN 二分类（变化 vs 稳定）；
        - 原始 mean-over-time 特征作为 baseline。
        """
        print(f">>> [ExpProbe] Starting KNN GlanceTraining Segmentation (k={n_neighbors})...")

        cache_key = f"GlanceTraining_Segmentation_{args.data}_{args.seq_len}_{getattr(args, 'sampling_stride', 'default')}"
        use_cached_baseline = cache_key in _baseline_cache

        _, data_loader = data_provider_GlanceTraining_Segmentation(
            args, flag="GlanceTraining_Segmentation", disable_ddp_split=True
        )
        print("args.num workers", args.num_workers)
        backbone_model = self._get_backbone(model)
        backbone_model.eval()
        backbone_model.to(self.device)

        preds_embed_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []

        if not use_cached_baseline:
            preds_raw_list: List[np.ndarray] = []

        with torch.no_grad():
            for batch in data_loader:
                batch_x = batch[0].float().to(self.device)
                batch_x_mark = batch[1].float().to(self.device)
                labels = batch[2].to(self.device)

                with autocast():
                    outputs = backbone_model.encode(batch_x, batch_x_mark, imputator=imputator)

                embedding = self._probe_cls_embedding(
                    outputs, args, seq_len=int(batch_x.shape[1])
                )
                preds_embed_list.append(embedding.detach().cpu().numpy())
                labels_list.append(labels.detach().cpu().numpy())

                if not use_cached_baseline:
                    batch_x_filled = torch.nan_to_num(batch_x, nan=0.0)
                    raw_feat = batch_x_filled.mean(dim=1)
                    preds_raw_list.append(raw_feat.detach().cpu().numpy())

        X_all_embed = np.concatenate(preds_embed_list, axis=0)
        y_all = np.concatenate(labels_list, axis=0).reshape(-1)
        del preds_embed_list, labels_list

        print(f">>> [GlanceTraining_Segmentation] Embedding features: X={X_all_embed.shape}, y={y_all.shape}")

        if use_cached_baseline:
            cached_data = _baseline_cache[cache_key]
            X_all_raw = cached_data["X_all_raw"]
            idx_train = cached_data["idx_train"]
            idx_test = cached_data["idx_test"]
            print(f">>> [Using cached baseline] Train: {len(idx_train)}, Test: {len(idx_test)}")
        else:
            X_all_raw = np.concatenate(preds_raw_list, axis=0)
            del preds_raw_list

            indices = np.arange(len(y_all))
            try:
                idx_train, idx_test = train_test_split(
                    indices,
                    train_size=0.8,
                    random_state=42,
                    stratify=y_all,
                )
            except ValueError:
                idx_train, idx_test = train_test_split(
                    indices,
                    train_size=0.8,
                    random_state=42,
                )

            _baseline_cache[cache_key] = {
                "X_all_raw": X_all_raw,
                "idx_train": idx_train,
                "idx_test": idx_test,
            }
            print(f">>> [Computing baseline] Train: {len(idx_train)}, Test: {len(idx_test)}")

        unique, counts = np.unique(y_all, return_counts=True)
        print(f">>> [GlanceTraining_Segmentation] Class Distribution: {dict(zip(unique, counts))}")

        self._eval_knn_metrics(
            X_all_embed[idx_train],
            y_all[idx_train],
            X_all_embed[idx_test],
            y_all[idx_test],
            n_neighbors,
            "Glance Segmentation Encoded CLS",
            average_method="binary",
        )

        if use_cached_baseline:
            cached_result = _baseline_cache[cache_key].get("baseline_result_glance_segmentation")
            if cached_result:
                print(cached_result, end="")
        else:
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            self._eval_knn_metrics(
                X_all_raw[idx_train],
                y_all[idx_train],
                X_all_raw[idx_test],
                y_all[idx_test],
                n_neighbors,
                "Glance Segmentation Raw Mean",
                average_method="binary",
            )
            baseline_result = buffer.getvalue()
            sys.stdout = old_stdout
            print(baseline_result, end="")
            _baseline_cache[cache_key]["baseline_result_glance_segmentation"] = baseline_result

        return {}

    def knn_probe_GlobalTree_Classification(self, model: torch.nn.Module, args: Any, n_neighbors: int = 5, imputator=None) -> Dict[str, float]:
        """
        GlobalTree 分类 KNN 探针（结构与 LCMAP Classification 对齐，并支持 AlphaEarth 结合）。
        """
        print(f">>> [ExpProbe] Starting KNN GlobalTree Classification (k={n_neighbors})...")
        cls_min_samples = int(getattr(args, 'cls_min_samples_per_class', 10))
        cls_few_shot_k = getattr(args, 'cls_few_shot_k', None)
        cache_key = (
            f"GlobalTree_Classification_{args.data}_{args.seq_len}_{getattr(args, 'sampling_stride', 'default')}"
            f"_min{cls_min_samples}_few{cls_few_shot_k or 'ratio'}"
        )
        use_cached_baseline = cache_key in _baseline_cache

        _, data_loader = data_provider_GlobalTree_Classification(args, flag='GlobalTree_Classification', disable_ddp_split=True)
        backbone_model = self._get_backbone(model)
        backbone_model.eval()
        backbone_model.to(self.device)

        preds_embed_list, preds_patch_avg_list, labels_list = [], [], []
        if not use_cached_baseline:
            preds_raw_list = []

        with torch.no_grad():
            for batch in data_loader:
                batch_x = batch[0].float().to(self.device)
                batch_x_mark = batch[1].float().to(self.device)
                labels = batch[2].to(self.device)

                with autocast():
                    outputs = backbone_model.encode(batch_x, batch_x_mark, imputator=imputator)

                embedding = self._probe_cls_embedding(
                    outputs, args, seq_len=int(batch_x.shape[1])
                )
                preds_embed_list.append(embedding.detach().cpu().numpy())
                labels_list.append(labels.detach().cpu().numpy())

                patch_tokens = outputs.get('patch_tokens', None)
                if patch_tokens is not None:
                    if patch_tokens.ndim == 3:
                        patch_avg = patch_tokens.mean(dim=1)
                    elif patch_tokens.ndim == 4:
                        patch_avg = patch_tokens.mean(dim=1).mean(dim=1)
                    else:
                        patch_avg = None
                    if patch_avg is not None:
                        preds_patch_avg_list.append(patch_avg.detach().cpu().numpy())

                if not use_cached_baseline:
                    batch_x_filled = torch.nan_to_num(batch_x, nan=0.0)
                    raw_feat = batch_x_filled.mean(dim=1)
                    preds_raw_list.append(raw_feat.detach().cpu().numpy())

        X_all_embed = np.concatenate(preds_embed_list, axis=0)
        y_all = np.concatenate(labels_list, axis=0).reshape(-1)
        del preds_embed_list, labels_list

        X_all_patch_avg = None
        if len(preds_patch_avg_list) > 0:
            try:
                X_all_patch_avg = np.concatenate(preds_patch_avg_list, axis=0)
                print(f">>> [GlobalTree] Patch-avg features: X={X_all_patch_avg.shape}")
            except Exception as e:
                print(f">>> [GlobalTree Warning] Failed to concatenate patch-avg features, skip patch-based combinations. Error: {e}")
                X_all_patch_avg = None
        preds_patch_avg_list = []

        print(f">>> [GlobalTree] Embedding features: X={X_all_embed.shape}, y={y_all.shape}")
        y_all_before_class_filter = y_all.copy()
        idx_keep, y_remap = _filter_small_classes(y_all, min_samples_per_class=cls_min_samples)
        if len(idx_keep) == 0:
            print(f">>> [GlobalTree] No samples after filtering classes with < {cls_min_samples} samples. Skip.")
            return {}
        X_all_embed = X_all_embed[idx_keep]
        y_all = y_remap
        if X_all_patch_avg is not None:
            X_all_patch_avg = X_all_patch_avg[idx_keep]
        print(f">>> [GlobalTree] After filtering small classes: N={len(y_all)}, n_class={len(np.unique(y_all))}")

        if use_cached_baseline:
            cached_data = _baseline_cache[cache_key]
            X_all_raw = cached_data['X_all_raw']
            idx_train = cached_data['idx_train']
            idx_test = cached_data['idx_test']
            print(f">>> [GlobalTree Using cached baseline] Train: {len(idx_train)}, Test: {len(idx_test)}")
        else:
            X_all_raw = np.concatenate(preds_raw_list, axis=0)
            del preds_raw_list
            X_all_raw = X_all_raw[idx_keep]
            cls_seed = int(getattr(args, 'cls_split_seed', 42))
            cls_ratio = float(getattr(args, 'cls_train_ratio', 0.8))
            if cls_few_shot_k is not None and cls_few_shot_k > 0:
                idx_train, idx_test = _get_few_shot_split(len(y_all), y_all, cls_few_shot_k, cls_seed)
                print(f">>> [GlobalTree Split] Few-shot k={cls_few_shot_k} per class, seed={cls_seed}")
            else:
                indices = np.arange(len(y_all))
                try:
                    idx_train, idx_test = train_test_split(
                        indices, train_size=cls_ratio, random_state=cls_seed, stratify=y_all,
                    )
                except ValueError:
                    idx_train, idx_test = train_test_split(indices, train_size=cls_ratio, random_state=cls_seed)
                print(f">>> [GlobalTree Split] train_ratio={cls_ratio}, seed={cls_seed}")

            _baseline_cache[cache_key] = {
                'X_all_raw': X_all_raw,
                'idx_train': idx_train,
                'idx_test': idx_test,
            }
            print(f">>> [GlobalTree Computing baseline] Train: {len(idx_train)}, Test: {len(idx_test)}")

        unique, counts = np.unique(y_all, return_counts=True)
        print(f">>> [GlobalTree] Class Distribution: {dict(zip(unique, counts))}")

        self._eval_knn_metrics(
            X_all_embed[idx_train],
            y_all[idx_train],
            X_all_embed[idx_test],
            y_all[idx_test],
            n_neighbors,
            "GlobalTree Encoded CLS",
            average_method='macro',
        )

        self._eval_knn_cls_plus_patch_avg(
            X_all_embed,
            X_all_patch_avg,
            idx_train,
            idx_test,
            y_all,
            n_neighbors,
            task_title_prefix="GlobalTree",
        )

        if use_cached_baseline:
            cached_result = _baseline_cache[cache_key].get('baseline_result_classification')
            if cached_result:
                print(cached_result, end='')
        else:
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            self._eval_knn_metrics(
                X_all_raw[idx_train],
                y_all[idx_train],
                X_all_raw[idx_test],
                y_all[idx_test],
                n_neighbors,
                "GlobalTree Raw Mean",
                average_method='macro',
            )
            baseline_result = buffer.getvalue()
            sys.stdout = old_stdout
            print(baseline_result, end='')
            _baseline_cache[cache_key]['baseline_result_classification'] = baseline_result

        # ============ AlphaEarth 结合 ============
        use_alphaearth = getattr(args, 'use_alphaearth', False)
        if use_alphaearth:
            print("\n>>> [AlphaEarth] Start AlphaEarth embeddings analysis for GlobalTree Classification...")
            alphaearth_path = getattr(args, 'alphaearth_path', None)
            if not alphaearth_path:
                _ds_root = getattr(args, 'downstream_data_root', '/intelnvme01/ziyun/DownStreamTasks')
                alphaearth_path = os.path.join(
                    _ds_root, 'downstream_classification_task', 'globaltree_gse_classification.npz'
                )

            try:
                ae_npz = np.load(alphaearth_path, allow_pickle=True)
                ae_data = ae_npz['data']       # (N, T_window, D)
                ae_data = np.nan_to_num(ae_data, nan=0.0, posinf=0.0, neginf=0.0)
                ae_labels = ae_npz['labels']
                ae_years_all = ae_npz['years']         # 形状 (Y,)
                ae_year_window = ae_npz['year_window'] # 形状 (N, 2)，指向 years 索引

                ae_data = _alphaearth_data_after_class_filter(
                    ae_data, idx_keep, "GlobalTree Classification + AlphaEarth"
                )
                y_all_alpha = y_all
                X_all_embed_alpha = X_all_embed
                X_all_raw_alpha = X_all_raw
                idx_train_alpha = idx_train
                idx_test_alpha = idx_test
                if X_all_patch_avg is not None:
                    X_all_patch_alpha = X_all_patch_avg
                else:
                    X_all_patch_alpha = None

                try:
                    ae_lab = np.asarray(ae_labels).reshape(-1)
                    if ae_lab.size > int(idx_keep.max()):
                        y_exp = y_all_before_class_filter[idx_keep]
                        if not np.array_equal(ae_lab[idx_keep].astype(y_exp.dtype, copy=False), y_exp):
                            print(
                                ">>> [AlphaEarth Warning] GlobalTree: ae_labels[idx_keep] != labels before class filter; "
                                "check npz vs dataloader order."
                            )
                except Exception:
                    pass

                # AlphaEarth 代表空间平均状态：如存在多年份窗口，统一在时间维上做平均得到单一向量。
                if ae_data.ndim == 3:
                    ae_avg = ae_data.mean(axis=1).astype(np.float32)
                elif ae_data.ndim == 2:
                    ae_avg = ae_data.astype(np.float32)
                else:
                    raise ValueError(f"Unexpected GlobalTree AlphaEarth data ndim={ae_data.ndim}, expected 2 or 3.")

                print(f">>> [AlphaEarth] GlobalTree AlphaEarth ae_avg shape = {ae_avg.shape}")

                from sklearn.preprocessing import StandardScaler

                def _l2_norm_rows(X: np.ndarray) -> np.ndarray:
                    norm = np.linalg.norm(X, axis=1, keepdims=True)
                    norm = np.where(norm < 1e-8, 1.0, norm)
                    return X / norm

                def build_concat_features(train_idx, test_idx, feat_list, name: str):
                    train_blocks, test_blocks = [], []
                    for feat in feat_list:
                        if feat is None:
                            continue
                        scaler = StandardScaler()
                        feat_train = scaler.fit_transform(feat[train_idx])
                        feat_test = scaler.transform(feat[test_idx])
                        train_blocks.append(feat_train)
                        test_blocks.append(feat_test)
                    if not train_blocks:
                        return None, None
                    X_train_cat = np.concatenate(train_blocks, axis=1)
                    X_test_cat = np.concatenate(test_blocks, axis=1)
                    print(f">>> [AlphaEarth] {name} features: X_train={X_train_cat.shape}, X_test={X_test_cat.shape}")
                    return X_train_cat, X_test_cat

                def build_concat_l2_blocks(train_idx, test_idx, feat_list, name: str):
                    train_blocks, test_blocks = [], []
                    for feat in feat_list:
                        if feat is None:
                            continue
                        train_blocks.append(_l2_norm_rows(feat[train_idx]))
                        test_blocks.append(_l2_norm_rows(feat[test_idx]))
                    if not train_blocks:
                        return None, None
                    X_train_cat = np.concatenate(train_blocks, axis=1)
                    X_test_cat = np.concatenate(test_blocks, axis=1)
                    print(f">>> [AlphaEarth] {name} (L2-norm blocks) features: X_train={X_train_cat.shape}, X_test={X_test_cat.shape}")
                    return X_train_cat, X_test_cat

                print(">>> [AlphaEarth] Baseline Encoded CLS accuracy is reported above (without AlphaEarth).")

                X_train_ae_only, X_test_ae_only = build_concat_features(
                    idx_train_alpha,
                    idx_test_alpha,
                    [ae_avg],
                    name="GlobalTree AlphaEarth Only",
                )
                if X_train_ae_only is not None:
                    self._eval_knn_metrics(
                        X_train_ae_only,
                        y_all_alpha[idx_train_alpha],
                        X_test_ae_only,
                        y_all_alpha[idx_test_alpha],
                        n_neighbors,
                        "GlobalTree AlphaEarthOnly",
                        average_method='macro',
                    )
                    self._eval_knn_metrics(
                        X_train_ae_only,
                        y_all_alpha[idx_train_alpha],
                        X_test_ae_only,
                        y_all_alpha[idx_test_alpha],
                        n_neighbors,
                        "GlobalTree AlphaEarthOnly (weights=distance)",
                        average_method='macro',
                        knn_weights='distance',
                    )

                X_train_ae_cls, X_test_ae_cls = build_concat_features(
                    idx_train_alpha,
                    idx_test_alpha,
                    [ae_avg, X_all_embed_alpha],
                    name="GlobalTree AlphaEarth + CLS",
                )
                if X_train_ae_cls is not None:
                    self._eval_knn_metrics(
                        X_train_ae_cls,
                        y_all_alpha[idx_train_alpha],
                        X_test_ae_cls,
                        y_all_alpha[idx_test_alpha],
                        n_neighbors,
                        "GlobalTree AlphaEarth+CLS",
                        average_method='macro',
                    )

                X_train_ae_cls_l2, X_test_ae_cls_l2 = build_concat_l2_blocks(
                    idx_train_alpha,
                    idx_test_alpha,
                    [ae_avg, X_all_embed_alpha],
                    name="GlobalTree AlphaEarth + CLS",
                )
                if X_train_ae_cls_l2 is not None:
                    self._eval_knn_metrics(
                        X_train_ae_cls_l2,
                        y_all_alpha[idx_train_alpha],
                        X_test_ae_cls_l2,
                        y_all_alpha[idx_test_alpha],
                        n_neighbors,
                        "GlobalTree AlphaEarth+CLS (L2-norm)",
                        average_method='macro',
                    )
                    self._eval_knn_metrics(
                        X_train_ae_cls_l2,
                        y_all_alpha[idx_train_alpha],
                        X_test_ae_cls_l2,
                        y_all_alpha[idx_test_alpha],
                        n_neighbors,
                        "GlobalTree AlphaEarth+CLS (L2-norm, weights=distance)",
                        average_method='macro',
                        knn_weights='distance',
                    )

                # 晚融合（与 LCMAP Classification 对齐，便于跨数据集对比）
                try:
                    max_jobs = min(8, os.cpu_count() or 4)
                    knn_ae = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=max_jobs)
                    knn_cls = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=max_jobs)
                    knn_ae.fit(X_train_ae_only, y_all_alpha[idx_train_alpha])
                    knn_cls.fit(X_all_embed_alpha[idx_train_alpha], y_all_alpha[idx_train_alpha])
                    proba_ae = knn_ae.predict_proba(X_test_ae_only)
                    proba_cls = knn_cls.predict_proba(X_all_embed_alpha[idx_test_alpha])
                    unified_classes = np.unique(y_all_alpha)

                    def reindex_proba(proba, classes):
                        cols = []
                        for c in unified_classes:
                            idx = np.where(classes == c)[0]
                            cols.append(proba[:, idx[0]] if len(idx) > 0 else np.zeros(proba.shape[0]))
                        return np.stack(cols, axis=1)

                    proba_ae = reindex_proba(proba_ae, knn_ae.classes_)
                    proba_cls = reindex_proba(proba_cls, knn_cls.classes_)
                    y_test_late = y_all_alpha[idx_test_alpha]
                    for w_ae, label in [
                        (0.5, "GlobalTree LateFusion AE+CLS (avg)"),
                        (0.6, "GlobalTree LateFusion AE+CLS (0.6 AE)"),
                        (0.65, "GlobalTree LateFusion AE+CLS (0.65 AE)"),
                        (0.7, "GlobalTree LateFusion AE+CLS (0.7 AE)"),
                    ]:
                        w_cls = 1.0 - w_ae
                        combined_proba = w_ae * proba_ae + w_cls * proba_cls
                        y_pred_late = np.argmax(combined_proba, axis=1)
                        print(f">>> KNN ({label}):")
                        print(
                            f"    Acc: {accuracy_score(y_test_late, y_pred_late):.4f} | "
                            f"Balanced Acc: {balanced_accuracy_score(y_test_late, y_pred_late):.4f} | "
                            f"F1-macro: {f1_score(y_test_late, y_pred_late, average='macro', zero_division=0):.4f} | AUC: N/A"
                        )
                    if X_all_patch_alpha is not None:
                        try:
                            knn_patch = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=max_jobs)
                            knn_patch.fit(X_all_patch_alpha[idx_train_alpha], y_all_alpha[idx_train_alpha])
                            proba_patch = knn_patch.predict_proba(X_all_patch_alpha[idx_test_alpha])
                            proba_patch = reindex_proba(proba_patch, knn_patch.classes_)
                            combined_3 = 0.5 * proba_ae + 0.3 * proba_cls + 0.2 * proba_patch
                            y_pred_3 = np.argmax(combined_3, axis=1)
                            print(f">>> KNN (GlobalTree LateFusion AE+CLS+Patch 0.5/0.3/0.2):")
                            print(
                                f"    Acc: {accuracy_score(y_test_late, y_pred_3):.4f} | "
                                f"Balanced Acc: {balanced_accuracy_score(y_test_late, y_pred_3):.4f} | "
                                f"F1-macro: {f1_score(y_test_late, y_pred_3, average='macro', zero_division=0):.4f} | AUC: N/A"
                            )
                        except Exception as e3:
                            print(f">>> [AlphaEarth] GlobalTree three-way late fusion failed: {e3}")
                except Exception as e_lf:
                    print(f">>> [AlphaEarth] GlobalTree late fusion failed: {e_lf}")

                if X_all_patch_alpha is not None:
                    X_train_ae_cls_patch, X_test_ae_cls_patch = build_concat_features(
                        idx_train_alpha,
                        idx_test_alpha,
                        [ae_avg, X_all_embed_alpha, X_all_patch_alpha],
                        name="GlobalTree AlphaEarth + CLS + PatchAvg",
                    )
                    if X_train_ae_cls_patch is not None:
                        self._eval_knn_metrics(
                            X_train_ae_cls_patch,
                            y_all_alpha[idx_train_alpha],
                            X_test_ae_cls_patch,
                            y_all_alpha[idx_test_alpha],
                            n_neighbors,
                            "GlobalTree AlphaEarth+CLS+PatchAvg",
                            average_method='macro',
                        )

                    X_train_ae_cls_patch_l2, X_test_ae_cls_patch_l2 = build_concat_l2_blocks(
                        idx_train_alpha,
                        idx_test_alpha,
                        [ae_avg, X_all_embed_alpha, X_all_patch_alpha],
                        name="GlobalTree AlphaEarth + CLS + PatchAvg",
                    )
                    if X_train_ae_cls_patch_l2 is not None:
                        self._eval_knn_metrics(
                            X_train_ae_cls_patch_l2,
                            y_all_alpha[idx_train_alpha],
                            X_test_ae_cls_patch_l2,
                            y_all_alpha[idx_test_alpha],
                            n_neighbors,
                            "GlobalTree AlphaEarth+CLS+PatchAvg (L2-norm)",
                            average_method='macro',
                        )
                else:
                    print(
                        ">>> [AlphaEarth] GlobalTree: patch-avg not available from encode; "
                        "skip AlphaEarth+CLS+PatchAvg early concat and three-way late fusion (already skipped above)."
                    )
            except Exception as e:
                print(f">>> [AlphaEarth Warning] GlobalTree AlphaEarth analysis failed: {e}")

        return {}

    def knn_probe_CDL_Classification(self, model: torch.nn.Module, args: Any, n_neighbors: int = 5, imputator=None) -> Dict[str, float]:
        """
        CDL 单年分类 KNN 探针（结构与 LCMAP Classification 对齐，并支持 AlphaEarth 结合）。
        """
        print(f">>> [ExpProbe] Starting KNN CDL Classification (k={n_neighbors})...")
        cls_min_samples = int(getattr(args, 'cls_min_samples_per_class', 10))
        cls_few_shot_k = getattr(args, 'cls_few_shot_k', None)
        cache_key = (
            f"CDL_Classification_{args.data}_{args.seq_len}_{getattr(args, 'sampling_stride', 'default')}"
            f"_min{cls_min_samples}_few{cls_few_shot_k or 'ratio'}"
        )
        use_cached_baseline = cache_key in _baseline_cache

        _, data_loader = data_provider_CDL_Classification(args, flag='CDL_Classification', disable_ddp_split=True)
        backbone_model = self._get_backbone(model)
        backbone_model.eval()
        backbone_model.to(self.device)

        preds_embed_list, preds_patch_avg_list, labels_list = [], [], []
        if not use_cached_baseline:
            preds_raw_list = []

        with torch.no_grad():
            for batch in data_loader:
                batch_x = batch[0].float().to(self.device)
                batch_x_mark = batch[1].float().to(self.device)
                labels = batch[2].to(self.device)

                with autocast():
                    outputs = backbone_model.encode(batch_x, batch_x_mark, imputator=imputator)

                embedding = self._probe_cls_embedding(
                    outputs, args, seq_len=int(batch_x.shape[1])
                )
                preds_embed_list.append(embedding.detach().cpu().numpy())
                labels_list.append(labels.detach().cpu().numpy())

                patch_tokens = outputs.get('patch_tokens', None)
                if patch_tokens is not None:
                    if patch_tokens.ndim == 3:
                        patch_avg = patch_tokens.mean(dim=1)
                    elif patch_tokens.ndim == 4:
                        patch_avg = patch_tokens.mean(dim=1).mean(dim=1)
                    else:
                        patch_avg = None
                    if patch_avg is not None:
                        preds_patch_avg_list.append(patch_avg.detach().cpu().numpy())

                if not use_cached_baseline:
                    batch_x_filled = torch.nan_to_num(batch_x, nan=0.0)
                    raw_feat = batch_x_filled.mean(dim=1)
                    preds_raw_list.append(raw_feat.detach().cpu().numpy())

        X_all_embed = np.concatenate(preds_embed_list, axis=0)
        y_all = np.concatenate(labels_list, axis=0).reshape(-1)
        del preds_embed_list, labels_list

        X_all_patch_avg = None
        if len(preds_patch_avg_list) > 0:
            try:
                X_all_patch_avg = np.concatenate(preds_patch_avg_list, axis=0)
                print(f">>> [CDL] Patch-avg features: X={X_all_patch_avg.shape}")
            except Exception as e:
                print(f">>> [CDL Warning] Failed to concatenate patch-avg features, skip patch-based combinations. Error: {e}")
                X_all_patch_avg = None
        preds_patch_avg_list = []

        print(f">>> [CDL] Embedding features: X={X_all_embed.shape}, y={y_all.shape}")
        y_all_before_class_filter = y_all.copy()
        idx_keep, y_remap = _filter_small_classes(y_all, min_samples_per_class=cls_min_samples)
        if len(idx_keep) == 0:
            print(f">>> [CDL] No samples after filtering classes with < {cls_min_samples} samples. Skip.")
            return {}
        X_all_embed = X_all_embed[idx_keep]
        y_all = y_remap
        if X_all_patch_avg is not None:
            X_all_patch_avg = X_all_patch_avg[idx_keep]
        print(f">>> [CDL] After filtering small classes: N={len(y_all)}, n_class={len(np.unique(y_all))}")

        # 若需要 t-SNE，保存 CDL 上 TED 的 CLS 表达
        if getattr(args, 'save_tsne_embeddings', False):
            try:
                from pathlib import Path
                out_dir = Path("logs")
                out_dir.mkdir(exist_ok=True)
                X_tsne, y_tsne = _sample_per_class_np(X_all_embed, y_all, max_per_class=20, seed=2026)
                np.savez(out_dir / "tsne_CDL_TED_embeddings.npz", X=X_tsne, y=y_tsne)
                print(f">>> [t-SNE] Saved CDL TED embeddings for t-SNE: {out_dir / 'tsne_CDL_TED_embeddings.npz'}")
            except Exception as e:
                print(f">>> [t-SNE Warning] Failed to save CDL TED embeddings: {e}")

        if use_cached_baseline:
            cached_data = _baseline_cache[cache_key]
            X_all_raw = cached_data['X_all_raw']
            idx_train = cached_data['idx_train']
            idx_test = cached_data['idx_test']
            print(f">>> [CDL Using cached baseline] Train: {len(idx_train)}, Test: {len(idx_test)}")
        else:
            X_all_raw = np.concatenate(preds_raw_list, axis=0)
            del preds_raw_list
            X_all_raw = X_all_raw[idx_keep]
            cls_seed = int(getattr(args, 'cls_split_seed', 42))
            cls_ratio = float(getattr(args, 'cls_train_ratio', 0.8))
            if cls_few_shot_k is not None and cls_few_shot_k > 0:
                idx_train, idx_test = _get_few_shot_split(len(y_all), y_all, cls_few_shot_k, cls_seed)
                print(f">>> [CDL Split] Few-shot k={cls_few_shot_k} per class, seed={cls_seed}")
            else:
                indices = np.arange(len(y_all))
                try:
                    idx_train, idx_test = train_test_split(
                        indices,
                        train_size=cls_ratio,
                        random_state=cls_seed,
                        stratify=y_all,
                    )
                except ValueError:
                    idx_train, idx_test = train_test_split(
                        indices,
                        train_size=cls_ratio,
                        random_state=cls_seed,
                    )
                print(f">>> [CDL Split] train_ratio={cls_ratio}, seed={cls_seed}")

            _baseline_cache[cache_key] = {
                'X_all_raw': X_all_raw,
                'idx_train': idx_train,
                'idx_test': idx_test,
            }
            print(f">>> [CDL Computing baseline] Train: {len(idx_train)}, Test: {len(idx_test)}")

        unique, counts = np.unique(y_all, return_counts=True)
        print(f">>> [CDL] Class Distribution: {dict(zip(unique, counts))}")

        self._eval_knn_metrics(
            X_all_embed[idx_train],
            y_all[idx_train],
            X_all_embed[idx_test],
            y_all[idx_test],
            n_neighbors,
            "CDL Encoded CLS",
            average_method='macro',
        )

        self._eval_knn_cls_plus_patch_avg(
            X_all_embed,
            X_all_patch_avg,
            idx_train,
            idx_test,
            y_all,
            n_neighbors,
            task_title_prefix="CDL",
        )

        if use_cached_baseline:
            cached_result = _baseline_cache[cache_key].get('baseline_result_classification')
            if cached_result:
                print(cached_result, end='')
        else:
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            self._eval_knn_metrics(
                X_all_raw[idx_train],
                y_all[idx_train],
                X_all_raw[idx_test],
                y_all[idx_test],
                n_neighbors,
                "CDL Raw Mean",
                average_method='macro',
            )
            baseline_result = buffer.getvalue()
            sys.stdout = old_stdout
            print(baseline_result, end='')
            _baseline_cache[cache_key]['baseline_result_classification'] = baseline_result

        # ============ AlphaEarth 结合 ============
        use_alphaearth = getattr(args, 'use_alphaearth', False)
        if use_alphaearth:
            print("\n>>> [AlphaEarth] Start AlphaEarth embeddings analysis for CDL Classification...")
            alphaearth_path = getattr(args, 'alphaearth_path', None)
            if not alphaearth_path:
                _ds_root = getattr(args, 'downstream_data_root', '/intelnvme01/ziyun/DownStreamTasks')
                alphaearth_path = os.path.join(
                    _ds_root, 'downstream_classification_task', 'cdl_gse_classification.npz'
                )

            try:
                ae_npz = np.load(alphaearth_path, allow_pickle=True)
                ae_data = ae_npz['data']
                ae_data = np.nan_to_num(ae_data, nan=0.0, posinf=0.0, neginf=0.0)
                ae_labels = ae_npz['labels']
                # year 是单个标量或长度为1的数组，这里仅作日志用途，不做筛选
                ae_year = ae_npz.get('year', None)

                ae_data = _alphaearth_data_after_class_filter(
                    ae_data, idx_keep, "CDL Classification + AlphaEarth"
                )
                y_all_alpha = y_all
                X_all_embed_alpha = X_all_embed
                X_all_raw_alpha = X_all_raw
                idx_train_alpha = idx_train
                idx_test_alpha = idx_test
                if X_all_patch_avg is not None:
                    X_all_patch_alpha = X_all_patch_avg
                else:
                    X_all_patch_alpha = None

                try:
                    ae_lab = np.asarray(ae_labels).reshape(-1)
                    if ae_lab.size > int(idx_keep.max()):
                        y_exp = y_all_before_class_filter[idx_keep]
                        if not np.array_equal(ae_lab[idx_keep].astype(y_exp.dtype, copy=False), y_exp):
                            print(
                                ">>> [AlphaEarth Warning] CDL: ae_labels[idx_keep] != labels before class filter; "
                                "check npz vs dataloader order."
                            )
                except Exception:
                    pass

                # 统一策略：AlphaEarth 代表空间平均状态，若存在多份时间/年份信息，直接在时间维度做平均。
                # 当前 CDL 文件 data 为 (N, D)，若未来扩展为 (N, T, D)，则在 T 维上平均。
                if ae_data.ndim == 3:
                    ae_avg = ae_data.mean(axis=1).astype(np.float32)
                elif ae_data.ndim == 2:
                    ae_avg = ae_data.astype(np.float32)
                else:
                    raise ValueError(f"Unexpected CDL AlphaEarth data ndim={ae_data.ndim}, expected 2 or 3.")
                print(f">>> [AlphaEarth] CDL AlphaEarth ae_avg shape = {ae_avg.shape}, year = {ae_year}")
                # 若需要 t-SNE，同样保存 CDL AlphaEarth 表达
                if getattr(args, 'save_tsne_embeddings', False):
                    try:
                        from pathlib import Path
                        out_dir = Path("logs")
                        out_dir.mkdir(exist_ok=True)
                        X_tsne_ae, y_tsne_ae = _sample_per_class_np(ae_avg, y_all_alpha, max_per_class=20, seed=2026)
                        np.savez(out_dir / "tsne_CDL_AlphaEarth_embeddings.npz", X=X_tsne_ae, y=y_tsne_ae)
                        print(f">>> [t-SNE] Saved CDL AlphaEarth embeddings for t-SNE: {out_dir / 'tsne_CDL_AlphaEarth_embeddings.npz'}")
                    except Exception as e:
                        print(f">>> [t-SNE Warning] Failed to save CDL AlphaEarth embeddings: {e}")

                from sklearn.preprocessing import StandardScaler

                def _l2_norm_rows(X: np.ndarray) -> np.ndarray:
                    norm = np.linalg.norm(X, axis=1, keepdims=True)
                    norm = np.where(norm < 1e-8, 1.0, norm)
                    return X / norm

                def build_concat_features(train_idx, test_idx, feat_list, name: str):
                    train_blocks, test_blocks = [], []
                    for feat in feat_list:
                        if feat is None:
                            continue
                        scaler = StandardScaler()
                        feat_train = scaler.fit_transform(feat[train_idx])
                        feat_test = scaler.transform(feat[test_idx])
                        train_blocks.append(feat_train)
                        test_blocks.append(feat_test)
                    if not train_blocks:
                        return None, None
                    X_train_cat = np.concatenate(train_blocks, axis=1)
                    X_test_cat = np.concatenate(test_blocks, axis=1)
                    print(f">>> [AlphaEarth] {name} features: X_train={X_train_cat.shape}, X_test={X_test_cat.shape}")
                    return X_train_cat, X_test_cat

                def build_concat_l2_blocks(train_idx, test_idx, feat_list, name: str):
                    train_blocks, test_blocks = [], []
                    for feat in feat_list:
                        if feat is None:
                            continue
                        train_blocks.append(_l2_norm_rows(feat[train_idx]))
                        test_blocks.append(_l2_norm_rows(feat[test_idx]))
                    if not train_blocks:
                        return None, None
                    X_train_cat = np.concatenate(train_blocks, axis=1)
                    X_test_cat = np.concatenate(test_blocks, axis=1)
                    print(f">>> [AlphaEarth] {name} (L2-norm blocks) features: X_train={X_train_cat.shape}, X_test={X_test_cat.shape}")
                    return X_train_cat, X_test_cat

                print(">>> [AlphaEarth] Baseline Encoded CLS accuracy is reported above (without AlphaEarth).")

                X_train_ae_only, X_test_ae_only = build_concat_features(
                    idx_train_alpha,
                    idx_test_alpha,
                    [ae_avg],
                    name="CDL AlphaEarth Only",
                )
                if X_train_ae_only is not None:
                    self._eval_knn_metrics(
                        X_train_ae_only,
                        y_all_alpha[idx_train_alpha],
                        X_test_ae_only,
                        y_all_alpha[idx_test_alpha],
                        n_neighbors,
                        "CDL AlphaEarthOnly",
                        average_method='macro',
                    )
                    self._eval_knn_metrics(
                        X_train_ae_only,
                        y_all_alpha[idx_train_alpha],
                        X_test_ae_only,
                        y_all_alpha[idx_test_alpha],
                        n_neighbors,
                        "CDL AlphaEarthOnly (weights=distance)",
                        average_method='macro',
                        knn_weights='distance',
                    )

                X_train_ae_cls, X_test_ae_cls = build_concat_features(
                    idx_train_alpha,
                    idx_test_alpha,
                    [ae_avg, X_all_embed_alpha],
                    name="CDL AlphaEarth + CLS",
                )
                if X_train_ae_cls is not None:
                    self._eval_knn_metrics(
                        X_train_ae_cls,
                        y_all_alpha[idx_train_alpha],
                        X_test_ae_cls,
                        y_all_alpha[idx_test_alpha],
                        n_neighbors,
                        "CDL AlphaEarth+CLS",
                        average_method='macro',
                    )

                X_train_ae_cls_l2, X_test_ae_cls_l2 = build_concat_l2_blocks(
                    idx_train_alpha,
                    idx_test_alpha,
                    [ae_avg, X_all_embed_alpha],
                    name="CDL AlphaEarth + CLS",
                )
                if X_train_ae_cls_l2 is not None:
                    self._eval_knn_metrics(
                        X_train_ae_cls_l2,
                        y_all_alpha[idx_train_alpha],
                        X_test_ae_cls_l2,
                        y_all_alpha[idx_test_alpha],
                        n_neighbors,
                        "CDL AlphaEarth+CLS (L2-norm)",
                        average_method='macro',
                    )
                    self._eval_knn_metrics(
                        X_train_ae_cls_l2,
                        y_all_alpha[idx_train_alpha],
                        X_test_ae_cls_l2,
                        y_all_alpha[idx_test_alpha],
                        n_neighbors,
                        "CDL AlphaEarth+CLS (L2-norm, weights=distance)",
                        average_method='macro',
                        knn_weights='distance',
                    )

                    if X_all_patch_alpha is not None:
                        X_train_ae_cls_patch, X_test_ae_cls_patch = build_concat_features(
                            idx_train_alpha,
                            idx_test_alpha,
                            [ae_avg, X_all_embed_alpha, X_all_patch_alpha],
                            name="CDL AlphaEarth + CLS + PatchAvg",
                        )
                        if X_train_ae_cls_patch is not None:
                            self._eval_knn_metrics(
                                X_train_ae_cls_patch,
                                y_all_alpha[idx_train_alpha],
                                X_test_ae_cls_patch,
                                y_all_alpha[idx_test_alpha],
                                n_neighbors,
                                "CDL AlphaEarth+CLS+PatchAvg",
                                average_method='macro',
                            )

                        X_train_ae_cls_patch_l2, X_test_ae_cls_patch_l2 = build_concat_l2_blocks(
                            idx_train_alpha,
                            idx_test_alpha,
                            [ae_avg, X_all_embed_alpha, X_all_patch_alpha],
                            name="CDL AlphaEarth + CLS + PatchAvg",
                        )
                        if X_train_ae_cls_patch_l2 is not None:
                            self._eval_knn_metrics(
                                X_train_ae_cls_patch_l2,
                                y_all_alpha[idx_train_alpha],
                                X_test_ae_cls_patch_l2,
                                y_all_alpha[idx_test_alpha],
                                n_neighbors,
                                "CDL AlphaEarth+CLS+PatchAvg (L2-norm)",
                                average_method='macro',
                            )
            except Exception as e:
                print(f">>> [AlphaEarth Warning] CDL AlphaEarth analysis failed: {e}")

        return {}
    