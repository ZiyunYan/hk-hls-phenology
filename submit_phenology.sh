#!/usr/bin/env bash
# EVI vs EVI2 comparison, then Zhang SOS/EOS on both gap-filled cubes.
#SBATCH --account=resilientia
#SBATCH --partition=resilientia
#SBATCH --qos=qos_resilientia
#SBATCH --job-name=hk-phenology
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/work/projects/resilientia/ziyun/hk-hls-phenology/data/phenology/train_logs/%x_%j.out
#SBATCH --error=/work/projects/resilientia/ziyun/hk-hls-phenology/data/phenology/train_logs/%x_%j.err

set -euo pipefail
REPO=/work/projects/resilientia/ziyun/hk-hls-phenology
PY=/work/home/ziyun2026/miniconda3/envs/timesfm/bin/python
export OMP_NUM_THREADS=64
export NUMBA_NUM_THREADS=64
export MKL_NUM_THREADS=64
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$REPO/data/phenology/train_logs"
cd "$REPO"
echo "start $(date) host=$(hostname)"

echo "=== compare EVI and EVI2 ==="
"$PY" -u compare_evi_evi2.py -n 3000 --seed 20261008 --smooth sg9

echo "=== HK cube ==="
"$PY" -u phenology_from_cube.py --src "$REPO/data/phenology/veg_filled.nc" --tag hk --index both --smooth sg9

echo "=== USA cube ==="
"$PY" -u phenology_from_cube.py --src "$REPO/data/phenology/veg_filled_usa.nc" --tag usa --index both --smooth sg9

echo "done $(date)"
