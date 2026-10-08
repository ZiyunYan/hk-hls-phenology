#!/usr/bin/env bash
# Fastest impute: copy once, then one fill worker per GPU, merge shards.
# For max speed, stop other jobs on GPUs 0-3 first (e.g. sl122 DDP).
set -euo pipefail
PY=/home/ziyun218/.conda/envs/timesfm/bin/python
SCRIPT=/home/ziyun218/pyprojects/hk_phenology/impute_vegetation.py
LOGDIR=/intelnvme03/ziyun218/hls_49QHE_hk/phenology/train_logs
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
