#!/usr/bin/env bash
# Fast smooth-method comparison (no full-grid phenology).
set -euo pipefail
PY=/home/ziyun218/.conda/envs/timesfm/bin/python
cd /home/ziyun218/pyprojects/hk_phenology
"$PY" -u try_smooth_methods.py -n 300 --seed 20261008
