#!/usr/bin/env bash
# Hold-out score, then fill the cleaned cube with sl122, sl244 and sl366.
# Phenology uses Savitzky–Golay (sg9) on each filled cube.
#SBATCH --account=resilientia
#SBATCH --partition=resilientia
#SBATCH --qos=qos_resilientia
#SBATCH --job-name=hk-windows
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h800:8
#SBATCH --cpus-per-task=200
#SBATCH --mem=800G
#SBATCH --time=04:00:00
#SBATCH --output=/work/projects/resilientia/ziyun/hk-hls-phenology/data/phenology/train_logs/%x_%j.out
#SBATCH --error=/work/projects/resilientia/ziyun/hk-hls-phenology/data/phenology/train_logs/%x_%j.err

set -euo pipefail
REPO=/work/projects/resilientia/ziyun/hk-hls-phenology
PY=/work/home/ziyun2026/miniconda3/envs/timesfm/bin/python
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMBA_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH="$REPO/imputator_ssl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$REPO/data/phenology/train_logs" "$REPO/data/phenology/window_compare"
cd "$REPO"
echo "start $(date) host=$(hostname)"
nvidia-smi -L

SRC="$REPO/data/phenology/veg_cube_bolton.nc"
TAIL="$REPO/imputator_ssl/checkpoints/HK-Imputator-optical6-topk5-s1-sl122/checkpoint.pth"

echo "=== hold-out ==="
pids=()
for seq in 122 244 366; do
  gpu=0
  if [ "$seq" = 244 ]; then gpu=1; fi
  if [ "$seq" = 366 ]; then gpu=2; fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u compare_impute_windows.py \
    --seq "$seq" --n 4096 --batch 1024 \
    --out "$REPO/data/phenology/window_compare/sl${seq}.json" &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
  echo "hold-out failed"
  exit 1
fi
"$PY" - << 'PY'
import json
from pathlib import Path
root = Path("/work/projects/resilientia/ziyun/hk-hls-phenology/data/phenology/window_compare")
rows = [json.loads((root / f"sl{seq}.json").read_text()) for seq in (122, 244, 366)]
rows.sort(key=lambda r: r["mean_band_mae_dn"])
print("rank by mean band MAE (DN), lower is better")
for r in rows:
    print(f"sl{r['seq']}: band MAE {r['mean_band_mae_dn']:.2f}  EVI MAE {r['evi_mae']:.4f}  held-out {r['held_out_steps']}")
PY

fill_one() {
  local seq=$1 shards=$2 offset=$3
  local ckpt="$REPO/imputator_ssl/checkpoints/HK-Imputator-optical6-topk5-s1-sl${seq}/checkpoint.pth"
  local dst="$REPO/data/phenology/veg_filled_sl${seq}.nc"
  local tail_args=()
  if [ "$seq" = 244 ]; then
    tail_args=(--tail-ckpt "$TAIL")
  fi
  echo "=== fill sl${seq} shards ${shards} offset ${offset} $(date) ==="
  "$PY" -u impute_fast.py --stage all \
    --src "$SRC" --dst "$dst" --seq "$seq" --ckpt "$ckpt" \
    --tag "sl${seq}" --n-shards "$shards" --gpu-offset "$offset" \
    --extract-workers 16 "${tail_args[@]}"
}

echo "=== full cubes ==="
fill_one 366 4 0 &
p366=$!
fill_one 244 2 4 &
p244=$!
fill_one 122 2 6 &
p122=$!
fail=0
wait "$p366" || fail=1
wait "$p244" || fail=1
wait "$p122" || fail=1
if [ "$fail" -ne 0 ]; then
  echo "fill failed"
  exit 1
fi

echo "=== phenology sg9 ==="
export OMP_NUM_THREADS=64
export NUMBA_NUM_THREADS=64
export MKL_NUM_THREADS=64
pids=()
for seq in 122 244 366; do
  "$PY" -u phenology_from_cube.py \
    --src "$REPO/data/phenology/veg_filled_sl${seq}.nc" \
    --tag "sl${seq}" --index both --smooth sg9 &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
  echo "phenology failed"
  exit 1
fi
echo "done $(date)"
