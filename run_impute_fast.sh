#!/usr/bin/env bash
# Fastest impute: copy once, then one fill worker per GPU, merge shards.
# For max speed, stop other jobs on GPUs 0-3 first (e.g. sl122 DDP).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HK_DATA_ROOT="${HK_DATA_ROOT:-$REPO/data}"
if [[ -z "${PY:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PY="$CONDA_PREFIX/bin/python"
  elif [[ -x "$HOME/miniconda3/envs/timesfm/bin/python" ]]; then
    PY="$HOME/miniconda3/envs/timesfm/bin/python"
  elif [[ -x "$HOME/.conda/envs/timesfm/bin/python" ]]; then
    PY="$HOME/.conda/envs/timesfm/bin/python"
  else
    PY=python3
  fi
fi
SCRIPT="$REPO/impute_vegetation.py"
LOGDIR="$HK_DATA_ROOT/phenology/train_logs"
NGPU="${NGPU:-4}"
BATCH="${BATCH:-8192}"
# Lower BATCH (e.g. 4096) if sharing GPUs with training.

mkdir -p "$LOGDIR"
echo "=== prepare $(date) ==="
"$PY" -u "$SCRIPT" --stage prepare 2>&1 | tee "$LOGDIR/impute_prepare.log"

echo "=== fill $NGPU shards batch=$BATCH $(date) ==="
pids=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES="$i" "$PY" -u "$SCRIPT" --stage fill \
    --shard-id "$i" --n-shards "$NGPU" --gpu 0 --batch-size "$BATCH" \
    2>&1 | tee "$LOGDIR/impute_fill_gpu${i}.log" &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "=== merge $(date) ==="
"$PY" -u "$SCRIPT" --stage merge --n-shards "$NGPU" 2>&1 | tee "$LOGDIR/impute_merge.log"
echo "=== done $(date) ==="
