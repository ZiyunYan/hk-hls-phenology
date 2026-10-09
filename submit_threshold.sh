#!/usr/bin/env bash
# 15% and 50% amplitude thresholds on the sl122 EVI cube.
#SBATCH --account=resilientia
#SBATCH --partition=resilientia
#SBATCH --qos=qos_resilientia
#SBATCH --job-name=hk-threshold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/projects/resilientia/ziyun/hk-hls-phenology/data/phenology/train_logs/%x_%j.out
#SBATCH --error=/work/projects/resilientia/ziyun/hk-hls-phenology/data/phenology/train_logs/%x_%j.err

set -euo pipefail
REPO=/work/projects/resilientia/ziyun/hk-hls-phenology
PY=/work/home/ziyun2026/miniconda3/envs/timesfm/bin/python
export OMP_NUM_THREADS=32
export NUMBA_NUM_THREADS=32
export MKL_NUM_THREADS=32
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$REPO/data/phenology/train_logs"
cd "$REPO"
echo "start $(date) host=$(hostname)"
"$PY" -u phenology_threshold.py --window 15 \
  --src "$REPO/data/phenology/veg_filled_sl122.nc" \
  --dst "$REPO/data/phenology/phenology_threshold_sl122_evi.nc"
"$PY" -u plot_threshold_maps.py
echo "done $(date)"
