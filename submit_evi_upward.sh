#!/usr/bin/env bash
# Upward EVI despike at the paper's 45-day gap. One process per spatial tile.
#SBATCH --account=resilientia
#SBATCH --partition=resilientia
#SBATCH --qos=qos_resilientia
#SBATCH --job-name=hk-evi-up45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=222
#SBATCH --mem=2000G
#SBATCH --time=02:00:00
#SBATCH --output=/work/projects/resilientia/ziyun/hk-hls-phenology/data/phenology/train_logs/%x_%j.out
#SBATCH --error=/work/projects/resilientia/ziyun/hk-hls-phenology/data/phenology/train_logs/%x_%j.err

set -euo pipefail
REPO=/work/projects/resilientia/ziyun/hk-hls-phenology
PY=/work/home/ziyun2026/miniconda3/envs/timesfm/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export CLEAN_WORKERS=64
mkdir -p "$REPO/data/phenology/train_logs"
cd "$REPO"
echo "start $(date) host=$(hostname) cpus=$(nproc)"
"$PY" -u apply_evi_upward.py
echo "done $(date)"
