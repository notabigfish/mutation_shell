#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
taskset -c 0-15 python scripts/train.py --config configs/c1000/base_v5_align_kabsch_exclude_4A.yaml 
taskset -c 0-15 python scripts/train.py --config configs/c1000/base_v5_align_kabsch_exclude_8A.yaml
taskset -c 0-15 python scripts/train.py --config configs/c1000/base_v5_align_kabsch_all.yaml
taskset -c 0-15 python scripts/train.py --config configs/c1000/base_v5_align_tmalign.yaml
