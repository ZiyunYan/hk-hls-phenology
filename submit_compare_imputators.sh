#!/usr/bin/env bash
# Compare the USA 7-band Imputator with the HK 6-band Imputator, then write
# the USA gap-filled cube. The node has 8 H800s; job 24388 holds 2 CPUs.
#SBATCH --account=resilientia
#SBATCH --partition=resilientia
#SBATCH --qos=qos_resilientia
#SBATCH --job-name=hk-usa-impute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h800:8
#SBATCH --cpus-per-task=222
#SBATCH --mem=2000G
#SBATCH --time=08:00:00
#SBATCH --output=/work/projects/resilientia/ziyun/hk-hls-phenology/data/phenology/train_logs/%x_%j.out
#SBATCH --error=/work/projects/resilientia/ziyun/hk-hls-phenology/data/phenology/train_logs/%x_%j.err

set -euo pipefail
REPO=/work/projects/resilientia/ziyun/hk-hls-phenology
USA=/work/projects/resilientia/ziyun/TimeSeries_SSL_USA
PY=/work/home/ziyun2026/miniconda3/envs/timesfm/bin/python
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$REPO/data/phenology/train_logs"
cd "$REPO"
echo "start $(date) host=$(hostname)"
nvidia-smi -L

echo "=== prepare sample ==="
"$PY" -u compare_imputators.py --stage prepare --n-tiles 24 --per-tile 96 --n-plot 100 --seed 20261008

echo "=== infer HK and USA ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO/imputator_ssl" "$PY" -u compare_imputators.py --stage infer --model hk --batch-size 512 &
hk_pid=$!
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$USA" "$PY" -u compare_imputators.py --stage infer --model usa --batch-size 512 &
usa_pid=$!
hk_status=0
usa_status=0
wait "$hk_pid" || hk_status=$?
wait "$usa_pid" || usa_status=$?
if [ "$hk_status" -ne 0 ] || [ "$usa_status" -ne 0 ]; then
  echo "infer failed hk=$hk_status usa=$usa_status"
  exit 1
fi

echo "=== report ==="
"$PY" -u compare_imputators.py --stage report

echo "=== full USA cube ==="
PYTHONPATH="$USA${PYTHONPATH:+:$PYTHONPATH}" "$PY" -u impute_usa_fast.py --stage all --extract-workers 32 --n-shards 8 --batch-size 0
echo "done $(date)"
