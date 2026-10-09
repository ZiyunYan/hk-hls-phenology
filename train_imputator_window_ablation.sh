#!/usr/bin/env bash
# Train 1-year (122) and 2-year (244) window imputators; same recipe as topk5_s1 @ 366.
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
ROOT="$REPO/imputator_ssl"
DATA="$HK_DATA_ROOT/imputator_hk"
LOGDIR="$HK_DATA_ROOT/phenology/train_logs"
mkdir -p "$LOGDIR"

run_one() {
  local MID=$1 SEQ=$2
  local LOG="$LOGDIR/${MID}.log"
  echo "==== $(date) start $MID seq_len=$SEQ ====" | tee -a "$LOG"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --nproc_per_node=4 run.py \
    --mode train \
    --model_id "$MID" \
    --model Transformer \
    --data HLS \
    --root_path "$DATA" \
    --enc_in 6 --dec_in 6 --c_out 6 \
    --mask_rate 0.8 \
    --seq_len "$SEQ" --label_len "$SEQ" --pred_len "$SEQ" \
    --sampling_stride "$SEQ" \
    --delay 16 \
    --train_epochs 60 \
    --batch_size 1024 \
    --num_workers 4 \
    --learning_rate 0.001 \
    --scaling_rule none \
    --d_model 256 --n_heads 8 --e_layers 6 --d_ff 1024 \
    --devices 0,1,2,3 --use_multi_gpu 1 \
    --freeze_last_layer_epochs 0 \
    --warmup_epochs 1 \
    --imp_rec_loss mse --imp_huber_delta 1.0 --imp_rec_alpha 1.0 \
    --imp_smooth_beta 1.0 --imp_smooth_mode dy2 \
    --imp_trim_topk_per_seq 5 --imp_trim_min_keep 8 \
    --probe 0 --ddp_find_unused_parameters 1 \
    --max_train_steps 1000000 \
    2>&1 | tee -a "$LOG"
  echo "==== $(date) done $MID ====" | tee -a "$LOG"
}

run_one Imputator_hk_optical6_topk5_s1_sl122 122
# sl244 runs on 10.21.17.99 (see /home/ziyun/logs/Imputator_hk_sl244_host99.log)
