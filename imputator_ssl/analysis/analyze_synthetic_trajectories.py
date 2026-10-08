"""
对 `synthetic_trajectories.nc` 的 TED/自监督 encoder CLS token 语义轨迹漂移分析。

数据假设
--------
NetCDF 文件包含变量：
- data: shape [steps=732, bands=7, samples=2000]
- path_id: shape [samples]，取值 {1,2,3}
- ratio_val: shape [samples]，取值 [0,1]
- split_step: shape [samples]（本脚本不强依赖，仅用于可选 sanity check）

任务目标
--------
1) 特征提取：对每个样本提取 CLS token 向量（D 维）
2) 降维：PCA 将 2000×D 降到 2D/3D
3) 可视化：按 path_id 分色系；ratio_val 作为颜色深浅（越接近 0 越“浅”，越接近 1 越“深”）
4) 统计：计算 CLS 到 “A 簇中心” 的欧氏距离与 ratio_val 的 Pearson 相关系数

使用示例
--------
python -m TimeSeries_SSL_USA.analysis.analyze_synthetic_trajectories \
  --dataset_path /intelnvme01/ziyun/USA_OUTPUT/huge_dataset/synthetic_trajectories.nc \
  --output_dir /intelnvme01/ziyun/USA_OUTPUT/semantic_drift_out_SYN \
  --model TED \
  --checkpoint_path /path/to/ted_checkpoint.pth \
  --seq_len 732 --patch_len 31 --stride 31 --d_model 384 --n_heads 6 --e_layers 12 --d_ff 1536 \
  --enc_in 7 --c_out 7 \
  --pca_dim 2 --batch_size 64 --device cuda

备注
----
- 本脚本复用工程内 TED 模型的 `encode` 推理接口：`out = model.encode(x, timeMark=..., imputator=None)`，
  并从 `out["cls_token"]` 读取 CLS 表征。
- 若你希望调用“你自定义的推理接口”，可以在 `encodeClsTokens` 内替换那一行调用逻辑即可。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import torch
except Exception as e:
    raise RuntimeError(
        "未检测到 PyTorch（import torch 失败）。请在包含 PyTorch 的 Python 环境中运行本脚本，"
        "例如激活你的 conda/venv 后再执行。原始错误："
        f"{e}"
    ) from e


def ensureProjectImports() -> None:
    """
    修复工程内“非包前缀”的绝对导入，例如 `from layers...` / `from utils...`。

    你的模型文件（如 `TimeSeries_SSL_USA/models/TED.py`）使用了 `from layers.xxx import ...`，
    其中 `layers/` 实际位于 `TimeSeries_SSL_USA/layers/`。当用 `python -m TimeSeries_SSL_USA...` 运行时，
    顶层并没有 `layers` 包，需将 `.../TimeSeries_SSL_USA` 加入 sys.path。
    """
    try:
        repoRoot = Path(__file__).resolve().parents[2]
        pkgRoot = repoRoot / "TimeSeries_SSL_USA"
        # 1) repoRoot: 允许 `import TimeSeries_SSL_USA`
        # 2) pkgRoot : 允许 `import layers/utils/...` 这种历史写法
        repoStr = str(repoRoot)
        pkgStr = str(pkgRoot)
        if repoStr not in sys.path:
            sys.path.insert(0, repoStr)
        if pkgStr not in sys.path:
            sys.path.insert(0, pkgStr)
    except Exception as e:
        raise RuntimeError(f"ensureProjectImports failed: {e}") from e


def safeSaveFigure(fig, path: str) -> None:
    """
    保存图像（失败不终止）。
    """
    try:
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"[Figure] saved: {path}")
    except Exception as e:
        print(f"[Figure] save failed {path}: {e}")


def readSyntheticTrajectories(datasetPath: str) -> Dict[str, np.ndarray]:
    """
    读取 synthetic NetCDF，并统一返回：
    - x: [N, T, C] float32
    - pathId: [N] int64
    - ratioVal: [N] float32
    - splitStep: [N] int64（若不存在则返回 -1）

    该函数优先使用 xarray，其次回退到 netCDF4。
    """
    if not os.path.exists(datasetPath):
        raise FileNotFoundError(datasetPath)

    try:
        import xarray as xr  # lazy import

        ds = xr.open_dataset(datasetPath)
        if "data" not in ds:
            raise KeyError("variable 'data' not found in dataset")
        data = ds["data"].values
        timeVar = ds["time"].values if "time" in ds else None
        pathId = ds["path_id"].values if "path_id" in ds else None
        ratioVal = ds["ratio_val"].values if "ratio_val" in ds else None
        splitStep = ds["split_step"].values if "split_step" in ds else None
    except Exception as eXr:
        try:
            from netCDF4 import Dataset  # lazy import

            with Dataset(datasetPath, "r") as f:
                data = np.asarray(f.variables["data"][:])
                timeVar = np.asarray(f.variables["time"][:]) if "time" in f.variables else None
                pathId = np.asarray(f.variables["path_id"][:]) if "path_id" in f.variables else None
                ratioVal = np.asarray(f.variables["ratio_val"][:]) if "ratio_val" in f.variables else None
                splitStep = np.asarray(f.variables["split_step"][:]) if "split_step" in f.variables else None
        except Exception as eNc:
            raise RuntimeError(f"failed to read NetCDF via xarray({eXr}) and netCDF4({eNc})") from eNc

    if data.ndim != 3:
        raise ValueError(f"data must be 3D, got shape={data.shape}")

    # 期望 (steps, bands, samples) -> 转 (samples, steps, bands)
    # 同时兼容用户可能保存为 (samples, steps, bands)
    if data.shape[0] == 732 and data.shape[2] == 2000:
        x = np.transpose(data, (2, 0, 1)).astype(np.float32, copy=False)
    elif data.shape[0] == 2000:
        x = data.astype(np.float32, copy=False)
    else:
        # 兜底：尝试识别 samples 维
        stepsDim = int(np.argmax(data.shape))
        if stepsDim == 0:
            # (samples, ?, ?) -> 假设已经是 (N,T,C)
            x = data.astype(np.float32, copy=False)
        else:
            # 默认按 (T,C,N)
            x = np.transpose(data, (2, 0, 1)).astype(np.float32, copy=False)

    n, t, c = x.shape
    if pathId is None or ratioVal is None:
        raise KeyError("path_id / ratio_val not found in dataset (required for grouping/analysis)")

    pathIdArr = np.asarray(pathId).reshape(-1).astype(np.int64, copy=False)
    ratioValArr = np.asarray(ratioVal).reshape(-1).astype(np.float32, copy=False)
    if splitStep is None:
        splitStepArr = np.full((n,), -1, dtype=np.int64)
    else:
        splitStepArr = np.asarray(splitStep).reshape(-1).astype(np.int64, copy=False)

    if pathIdArr.shape[0] != n or ratioValArr.shape[0] != n:
        raise ValueError(
            f"metadata length mismatch: N={n}, path_id={pathIdArr.shape}, ratio_val={ratioValArr.shape}"
        )

    if not (0.0 <= float(np.nanmin(ratioValArr)) and float(np.nanmax(ratioValArr)) <= 1.0):
        print("[Warn] ratio_val out of [0,1] range, will still proceed.")

    if timeVar is None:
        raise KeyError("variable 'time' not found in dataset (required for timeMark)")
    timeArr = np.asarray(timeVar).reshape(-1)
    if timeArr.shape[0] != x.shape[1]:
        raise ValueError(f"time length mismatch: time={timeArr.shape[0]}, steps={x.shape[1]}")

    return {"x": x, "time": timeArr, "pathId": pathIdArr, "ratioVal": ratioValArr, "splitStep": splitStepArr}


def buildTimeMarkFromTime(timeArr: np.ndarray, freq: str = "RS", timeenc: int = 1) -> np.ndarray:
    """
    仿照 `data_loader.py` 的逻辑，从 `time` 构建 `data_stamp`。

    - data_loader: df_stamp = f.variables['time'][:].astype(str)
                  df_stamp = pd.to_datetime(df_stamp, format='%Y%j') 或自动推断
                  data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)

    Returns
    -------
    np.ndarray
        [T, F]，其中 freq='RS' 时 F=2 (DayOfYearSin/Cos)，与 TED 的 time_mark 期望一致。
    """
    import pandas as pd

    from utils.timefeatures import time_features

    if timeenc != 1:
        # 仍返回一个占位，但 TED 期望 2 维；这里给 2 维零向量
        return np.zeros((timeArr.shape[0], 2), dtype=np.float32)

    # 与 dataloader 保持一致：先转 string，再尝试 %Y%j（年+积日）格式；失败则自动推断
    try:
        dfStamp = timeArr.astype(str)
    except Exception:
        dfStamp = np.asarray(timeArr, dtype=str)

    try:
        dtIndex = pd.to_datetime(dfStamp, format="%Y%j")
    except Exception:
        dtIndex = pd.to_datetime(dfStamp)

    dataStamp = time_features(dtIndex, freq=freq).transpose(1, 0).astype(np.float32, copy=False)
    return dataStamp


def getDefaultPreScaler() -> Dict[str, np.ndarray]:
    """
    与 `TimeSeries_SSL_USA/data_provider/data_loader.py` 中的 hard-coded `pre_scaler` 对齐。

    Returns
    -------
    dict with keys: mean(std) both shape [7]
    """
    mean = np.array(
        [4.6641504e02, 7.3672772e02, 8.1836304e02, 2.6322727e03, 2.2375674e03, 1.4864814e03, 4.9718586e-01],
        dtype=np.float32,
    )
    std = np.array(
        [3.0122498e02, 3.7350461e02, 5.3584772e02, 1.0549794e03, 8.8861761e02, 8.3431287e02, 2.7525917e-01],
        dtype=np.float32,
    )
    return {"mean": mean, "std": std}


def applyPreScaling(x: np.ndarray, preScaler: Dict[str, np.ndarray]) -> np.ndarray:
    """
    按 band 做 z-score： (x - mean) / std

    Parameters
    ----------
    x:
        [N,T,C]
    preScaler:
        dict with mean/std shape [C]
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [N,T,C], got shape={x.shape}")
    mean = np.asarray(preScaler["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(preScaler["std"], dtype=np.float32).reshape(1, 1, -1)
    if mean.shape[-1] != x.shape[-1] or std.shape[-1] != x.shape[-1]:
        raise ValueError(f"preScaler dim mismatch: mean/std={mean.shape[-1]}, x C={x.shape[-1]}")
    return (x.astype(np.float32, copy=False) - mean) / std


@dataclass
class ModelConfigs:
    """
    构建模型需要的最小 configs（对齐工程里 models 的构造参数风格）。
    """

    model: str
    seq_len: int
    patch_len: int
    stride: int
    d_model: int
    n_heads: int
    e_layers: int
    d_ff: int
    enc_in: int
    c_out: int
    dropout: float = 0.0
    fc_dropout: float = 0.0
    head_dropout: float = 0.0
    num_register_tokens: int = 4
    mlp_ratio: int = 2
    use_gpu: int = 1
    use_multi_gpu: int = 0
    devices: str = "0"
    gpu: int = 0
    local_rank: int = 0
    imputator_mode: str = "full"


def buildModel(configs: ModelConfigs) -> torch.nn.Module:
    """
    根据 model 名称构建模型实例。
    """
    try:
        if configs.model == "TED":
            from TimeSeries_SSL_USA.models import TED as tedModule

            return tedModule.Model(configs).float()
        raise ValueError(f"unsupported model: {configs.model}")
    except Exception as e:
        raise RuntimeError(f"buildModel failed for {configs.model}: {e}") from e


def loadCheckpoint(model: torch.nn.Module, checkpointPath: str, device: torch.device) -> None:
    """
    加载 state_dict（兼容 strict=False，兼容 head 尺寸不匹配的过滤加载）。
    """
    try:
        if not os.path.exists(checkpointPath):
            raise FileNotFoundError(checkpointPath)
        ckpt = torch.load(checkpointPath, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
        except RuntimeError as e:
            print(f"[Checkpoint] initial load failed, filtering incompatible keys: {e}")
            modelState = model.state_dict()
            filteredState = {}
            for k, v in state.items():
                if k in modelState and isinstance(v, torch.Tensor) and isinstance(modelState[k], torch.Tensor):
                    if v.shape == modelState[k].shape:
                        filteredState[k] = v
                else:
                    filteredState[k] = v
            missing, unexpected = model.load_state_dict(filteredState, strict=False)

        model.to(device)
        model.eval()
        if len(unexpected) > 0:
            print(f"[Checkpoint] Unexpected keys: {unexpected[:10]} (total={len(unexpected)})")
        if len(missing) > 0:
            print(f"[Checkpoint] Missing keys: {missing[:10]} (total={len(missing)})")
    except Exception as e:
        raise RuntimeError(f"loadCheckpoint failed: {e}") from e


def resolveCheckpointPath(checkpointPath: str) -> str:
    """
    兼容传入目录（如 `./checkpoints/TED`）的情况：自动解析到具体 .pth 文件。
    """
    p = Path(checkpointPath)

    # 若用户传入相对路径但当前 cwd 不同，尝试基于仓库根目录与 TimeSeries_SSL_USA/ 目录再解析一次
    if not p.exists() and not p.is_absolute():
        try:
            repoRoot = Path(__file__).resolve().parents[2]
            pkgRoot = repoRoot / "TimeSeries_SSL_USA"
            alt1 = (repoRoot / p).resolve()
            alt2 = (pkgRoot / p).resolve()
            if alt1.exists():
                p = alt1
            elif alt2.exists():
                p = alt2
        except Exception:
            pass

    if p.is_file():
        return str(p)
    if p.is_dir():
        candidates = [
            "checkpoint.pth",
            "ckpt.pth",
            "model.pth",
            "best.pth",
            "best_model.pth",
            "last.pth",
        ]
        for name in candidates:
            cand = p / name
            if cand.exists() and cand.is_file():
                return str(cand)
        raise FileNotFoundError(
            f"checkpointPath is a directory but no known ckpt file found under it: {checkpointPath}"
        )
    raise FileNotFoundError(f"checkpointPath not found: {checkpointPath}")


@torch.no_grad()
def encodeClsTokens(
    model: torch.nn.Module,
    x: torch.Tensor,
    timeMark: Optional[torch.Tensor] = None,
    batchSize: int = 64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    批量调用推理接口提取 `cls_token`。

    Parameters
    ----------
    model:
        具有 `encode(xEnc, timeMark=..., imputator=None)` 方法的模型
    x:
        [N,T,C] float tensor
    timeMark:
        [N,T,2] 或 None。若 None，将在内部补零。
    """
    if device is None:
        device = x.device

    n = int(x.shape[0])
    clsList: List[torch.Tensor] = []
    for i in range(0, n, int(batchSize)):
        xb = x[i : i + batchSize].to(device, non_blocking=True)
        if timeMark is None:
            tb = torch.zeros(xb.shape[0], xb.shape[1], 2, device=device, dtype=xb.dtype)
        else:
            tb = timeMark[i : i + batchSize].to(device, non_blocking=True)

        out = model.encode(xb, timeMark=tb, imputator=None)
        cls = out["cls_token"]
        clsList.append(cls.detach().float().cpu())

    return torch.cat(clsList, dim=0)


def pcaProject(x: np.ndarray, pcaDim: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA 投影。

    Returns
    -------
    (z, explainedVarRatio)
    """
    if pcaDim not in (2, 3):
        raise ValueError("pcaDim must be 2 or 3")
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=pcaDim, random_state=42)
        z = pca.fit_transform(x)
        evr = pca.explained_variance_ratio_
        return z.astype(np.float32), evr.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"PCA failed: {e}") from e


def pearsonCorr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Pearson 相关系数与 p-value。
    优先使用 scipy；若不可用则回退到 numpy（p-value 返回 NaN）。
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        return float("nan"), float("nan")

    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        r = float(np.corrcoef(x, y)[0, 1])
        return r, float("nan")


def computeClusterCenters(
    clsTokens: np.ndarray,
    ratioVal: np.ndarray,
    pathId: np.ndarray,
    aRatioMax: float = 0.1,
    bRatioMin: float = 0.9,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    按 path_id 计算 A/B 簇中心：
    - A 簇：ratio_val <= aRatioMax
    - B 簇：ratio_val >= bRatioMin
    """
    centers: Dict[int, Dict[str, np.ndarray]] = {}
    for pid in sorted(set(int(v) for v in np.unique(pathId))):
        maskP = pathId == pid
        aMask = maskP & (ratioVal <= float(aRatioMax))
        bMask = maskP & (ratioVal >= float(bRatioMin))

        if aMask.sum() == 0:
            raise RuntimeError(f"no A-cluster samples for path_id={pid} with ratio_val <= {aRatioMax}")
        if bMask.sum() == 0:
            raise RuntimeError(f"no B-cluster samples for path_id={pid} with ratio_val >= {bRatioMin}")

        aCenter = clsTokens[aMask].mean(axis=0)
        bCenter = clsTokens[bMask].mean(axis=0)
        centers[int(pid)] = {"aCenter": aCenter.astype(np.float32), "bCenter": bCenter.astype(np.float32)}
    return centers


def main() -> None:
    ensureProjectImports()
    parser = argparse.ArgumentParser(description="Analyze CLS semantic drift on synthetic trajectories (NetCDF).")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=2000)

    parser.add_argument("--model", type=str, default="TED", choices=["TED"])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--patch_len", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--e_layers", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--enc_in", type=int, required=True)
    parser.add_argument("--c_out", type=int, required=True)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--pca_dim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--a_ratio_max", type=float, default=0.1, help="A cluster definition: ratio_val <= this")
    parser.add_argument("--b_ratio_min", type=float, default=0.9, help="B cluster definition: ratio_val >= this")
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="是否按 data_loader 的 pre_scaler 做标准化 (1=是,0=否)。默认=1",
    )
    parser.add_argument("--timeenc", type=int, default=1, help="是否启用 time features (与 data_loader 对齐，默认=1)")
    parser.add_argument("--freq", type=str, default="RS", help="time_features 的频率字符串（建议 RS，输出2维）")
    parser.add_argument(
        "--trunc_steps",
        type=str,
        default="",
        help="可选：对多个截断长度重复分析。格式示例：'122,244,366,488,610'。空字符串表示只跑完整长度。",
    )
    parser.add_argument(
        "--trunc_mode",
        type=str,
        default="tail",
        choices=["tail", "head"],
        help="截断方式：tail=取序列末端 L 步；head=取序列开头 L 步。",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # 1) read dataset
    ds = readSyntheticTrajectories(args.dataset_path)
    x = ds["x"][: int(args.max_samples)]
    timeArr = ds["time"]
    pathId = ds["pathId"][: int(args.max_samples)]
    ratioVal = ds["ratioVal"][: int(args.max_samples)]
    splitStep = ds["splitStep"][: int(args.max_samples)]

    n, t, c = x.shape
    print(f"[Data] x shape={x.shape}, path_id unique={sorted(set(pathId.tolist()))}")

    if c != int(args.enc_in):
        print(f"[Warn] enc_in={args.enc_in} but data bands={c}. Will still run, but模型可能不匹配。")

    # 与训练/推理保持一致的标准化（参考 data_loader.py 的 seq_x 处理）
    if int(args.scale) == 1:
        preScaler = getDefaultPreScaler()
        x = applyPreScaling(x, preScaler)
        print("[Data] applied pre_scaler z-score normalization (aligned with data_loader.py)")
    else:
        print("[Data] scale=0, skip normalization")

    # 与 data_loader.py 对齐的 timeMark: [T,2] for freq=RS
    baseTimeStamp = buildTimeMarkFromTime(timeArr, freq=str(args.freq), timeenc=int(args.timeenc))
    if baseTimeStamp.ndim != 2 or baseTimeStamp.shape[0] != x.shape[1]:
        raise RuntimeError(f"invalid timeMark shape from time: {baseTimeStamp.shape}, expected [T,F] with T={x.shape[1]}")
    if baseTimeStamp.shape[1] != 2:
        print(f"[Warn] timeMark feature dim={baseTimeStamp.shape[1]} (TED 通常期望2). freq={args.freq}")

    # 2) build model + ckpt
    configs = ModelConfigs(
        model=args.model,
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        enc_in=args.enc_in,
        c_out=args.c_out,
        dropout=args.dropout,
    )
    model = buildModel(configs)
    ckptPath = resolveCheckpointPath(args.checkpoint_path)
    loadCheckpoint(model, ckptPath, device=device)

    def parseTruncSteps(spec: str) -> List[int]:
        if spec.strip() == "":
            return []
        out: List[int] = []
        for part in spec.split(","):
            p = part.strip()
            if p == "":
                continue
            v = int(float(p))
            if v <= 0:
                continue
            out.append(v)
        # 去重并排序
        out = sorted(set(out))
        return out

    truncSteps = parseTruncSteps(args.trunc_steps)
    if len(truncSteps) == 0:
        truncSteps = [int(args.seq_len)]

    for truncLen in truncSteps:
        if truncLen > t:
            print(f"[Warn] truncLen={truncLen} > seriesLen={t}, will use full length {t}.")
            truncLenUse = int(t)
        else:
            truncLenUse = int(truncLen)

        subDir = args.output_dir
        # 若是 sweep（多个长度），按长度建子目录避免覆盖
        if len(truncSteps) > 1:
            subDir = os.path.join(args.output_dir, f"L{truncLenUse}")
        os.makedirs(subDir, exist_ok=True)

        # 3) truncate (head/tail), no padding (支持变长推理)
        if args.trunc_mode == "tail":
            xTrunc = x[:, -truncLenUse:, :]
            tTrunc = baseTimeStamp[-truncLenUse:, :]
        else:
            xTrunc = x[:, :truncLenUse, :]
            tTrunc = baseTimeStamp[:truncLenUse, :]

        xTensor = torch.from_numpy(xTrunc.astype(np.float32, copy=False)).float()
        timeTensor = torch.from_numpy(np.tile(tTrunc[None, :, :], (n, 1, 1))).float()
        clsTokens = encodeClsTokens(model, xTensor, timeMark=timeTensor, batchSize=args.batch_size, device=device).numpy()
        print(f"[Encode] truncLen={truncLenUse}, clsTokens shape={clsTokens.shape}, outDir={subDir}")

        # 3b) save raw embeddings package (as requested)
        rawNpzPath = os.path.join(subDir, "synthetic_cls_raw.npz")
        try:
            np.savez_compressed(
                rawNpzPath,
                clsEmbeddings=clsTokens.astype(np.float32),
                sampleIdx=np.arange(n, dtype=np.int64),
                pathId=pathId.astype(np.int64),
                ratioVal=ratioVal.astype(np.float32),
                splitStep=splitStep.astype(np.int64),
                truncLen=np.asarray([truncLenUse], dtype=np.int64),
                truncMode=np.asarray([args.trunc_mode], dtype=object),
            )
            print(f"[Done] saved raw embeddings package: {rawNpzPath}")
        except Exception as e:
            raise RuntimeError(f"failed to save npz to {rawNpzPath}: {e}") from e

        # 4) PCA
        z, evr = pcaProject(clsTokens, pcaDim=int(args.pca_dim))
        print(f"[PCA] truncLen={truncLenUse}, dim={args.pca_dim}, explained_var_ratio={evr.tolist()}")

        # 5) cluster centers + distance
        centers = computeClusterCenters(
            clsTokens=clsTokens,
            ratioVal=ratioVal,
            pathId=pathId,
            aRatioMax=float(args.a_ratio_max),
            bRatioMin=float(args.b_ratio_min),
        )
        distToA = np.zeros((n,), dtype=np.float32)
        distToB = np.zeros((n,), dtype=np.float32)
        for pid, cb in centers.items():
            mask = pathId == pid
            aCenter = cb["aCenter"][None, :]
            bCenter = cb["bCenter"][None, :]
            distToA[mask] = np.linalg.norm(clsTokens[mask] - aCenter, axis=1)
            distToB[mask] = np.linalg.norm(clsTokens[mask] - bCenter, axis=1)

        # 6) correlation (overall + per path)
        overallR, overallP = pearsonCorr(distToA, ratioVal)
        print(
            f"[Pearson] truncLen={truncLenUse} overall corr(dist_to_A, ratio_val) = r={overallR:.4f}, p={overallP:.3g}"
        )
        perPath = []
        for pid in sorted(centers.keys()):
            mask = pathId == pid
            r, p = pearsonCorr(distToA[mask], ratioVal[mask])
            perPath.append({"path_id": int(pid), "pearson_r": r, "p_value": p, "n": int(mask.sum())})
            print(f"[Pearson] truncLen={truncLenUse} path_id={pid}: r={r:.4f}, p={p:.3g}, n={int(mask.sum())}")

        # 7) save table
        cols = {}
        if z.shape[1] >= 2:
            cols["pca1"] = z[:, 0]
            cols["pca2"] = z[:, 1]
        if z.shape[1] == 3:
            cols["pca3"] = z[:, 2]
        outDf = pd.DataFrame(
            {
                "sample_idx": np.arange(n, dtype=np.int64),
                "path_id": pathId.astype(np.int64),
                "ratio_val": ratioVal.astype(np.float32),
                "split_step": splitStep.astype(np.int64),
                "dist_to_a": distToA.astype(np.float32),
                "dist_to_b": distToB.astype(np.float32),
                "trunc_len": np.full((n,), truncLenUse, dtype=np.int64),
                "trunc_mode": np.full((n,), args.trunc_mode, dtype=object),
                **cols,
            }
        )
        outCsv = os.path.join(subDir, "synthetic_cls_pca_metrics.csv")
        outDf.to_csv(outCsv, index=False)
        print(f"[Done] saved metrics: {outCsv}")

        outNpz = os.path.join(subDir, "synthetic_cls_embeddings.npz")
        try:
            np.savez_compressed(
                outNpz,
                clsTokens=clsTokens.astype(np.float32),
                pca=z.astype(np.float32),
                pathId=pathId.astype(np.int64),
                ratioVal=ratioVal.astype(np.float32),
                splitStep=splitStep.astype(np.int64),
                explainedVarRatio=evr.astype(np.float32),
                truncLen=np.asarray([truncLenUse], dtype=np.int64),
                truncMode=np.asarray([args.trunc_mode], dtype=object),
            )
            print(f"[Done] saved embeddings: {outNpz}")
        except Exception as e:
            print(f"[Warn] failed to save npz: {e}")

        # 8) figures
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
        except Exception as e:
            print(f"[Figure] matplotlib not available: {e}")
            continue

        # path -> colormap
        # 你当前 synthetic 数据为 5-path（1..5）。这里给每条路径一个稳定的色系映射。
        cmapByPath = {
            1: plt.cm.Greens,
            2: plt.cm.Oranges,
            3: plt.cm.Reds,
            4: plt.cm.Blues,
            5: plt.cm.Purples,
        }
        norm = Normalize(vmin=0.0, vmax=1.0, clip=True)

        fig = plt.figure(figsize=(8.0, 6.8))
        ax = fig.add_subplot(111)
        for pid in sorted(centers.keys()):
            mask = pathId == pid
            cmap = cmapByPath.get(pid, plt.cm.viridis)
            colors = cmap(norm(ratioVal[mask]))
            ax.scatter(
                z[mask, 0],
                z[mask, 1],
                s=22,
                c=colors,
                edgecolors="none",
                label=f"path {pid}",
                alpha=0.90,
            )

            aMask = mask & (ratioVal <= float(args.a_ratio_max))
            bMask = mask & (ratioVal >= float(args.b_ratio_min))
            if aMask.sum() > 0:
                ax.scatter(
                    z[aMask, 0].mean(),
                    z[aMask, 1].mean(),
                    marker="X",
                    s=120,
                    c=[cmap(0.20)],
                    linewidths=0.5,
                )
            if bMask.sum() > 0:
                ax.scatter(
                    z[bMask, 0].mean(),
                    z[bMask, 1].mean(),
                    marker="X",
                    s=120,
                    c=[cmap(0.90)],
                    linewidths=0.5,
                )

        ax.set_title(
            f"CLS PCA-{args.pca_dim} (truncLen={truncLenUse}, mode={args.trunc_mode})\n"
            f"overall Pearson r(dist_to_A, ratio_val)={overallR:.3f}"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False, markerscale=1.2)
        ax.grid(True, linestyle="--", alpha=0.25)
        safeSaveFigure(fig, os.path.join(subDir, f"synthetic_cls_pca2d_{args.model}.png"))
        plt.close(fig)

        if int(args.pca_dim) == 3:
            fig3 = plt.figure(figsize=(9.0, 7.2))
            ax3 = fig3.add_subplot(111, projection="3d")
            for pid in sorted(centers.keys()):
                mask = pathId == pid
                cmap = cmapByPath.get(pid, plt.cm.viridis)
                colors = cmap(norm(ratioVal[mask]))
                ax3.scatter(z[mask, 0], z[mask, 1], z[mask, 2], s=18, c=colors, edgecolors="none", alpha=0.90)
            ax3.set_title(f"CLS PCA-3D (truncLen={truncLenUse}, mode={args.trunc_mode}) | {args.model}")
            ax3.set_xlabel("PC1")
            ax3.set_ylabel("PC2")
            ax3.set_zlabel("PC3")
            safeSaveFigure(fig3, os.path.join(subDir, f"synthetic_cls_pca3d_{args.model}.png"))
            plt.close(fig3)

        corrDf = pd.DataFrame(perPath)
        corrDf.loc[len(corrDf)] = {"path_id": -1, "pearson_r": overallR, "p_value": overallP, "n": int(n)}
        corrCsv = os.path.join(subDir, "synthetic_cls_corr_summary.csv")
        corrDf.to_csv(corrCsv, index=False)
        print(f"[Done] saved correlation summary: {corrCsv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"[Error] {e}")
        traceback.print_exc()
