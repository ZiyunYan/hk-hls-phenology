#!/usr/bin/env bash
# Fill Hong Kong vegetation gaps with all 8 H800s.
# The node has 224 CPUs and 2063936 MB. Job 24385 currently holds 2 CPUs and 8 GB,
# so this asks for everything else and starts immediately.
#SBATCH --account=resilientia
#SBATCH --partition=resilientia
#SBATCH --qos=qos_resilientia
#SBATCH --job-name=hk-hls-impute
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
PY=/work/home/ziyun2026/miniconda3/envs/timesfm/bin/python
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$REPO/data/phenology/train_logs"
cd "$REPO"
echo "start $(date) host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-} gpus=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L
"$PY" -u impute_fast.py --stage all --extract-workers 32 --n-shards 8 --batch-size 0
echo "done $(date)"
