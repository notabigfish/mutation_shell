#!/usr/bin/env bash
set -euo pipefail

python scripts/create_time_split.py \
    --config configs/c1000/base_v5.yaml \
    --csv data/SingleMutPairs2024_subset_c1000.csv \
    --out data/processed/splits_time_c1000_2022_2023_2024.json \
    --cluster-policy latest_release \
    --train-max-year 2022 \
    --valid-year 2023 \
    --test-min-year 2024

python scripts/train.py \
    --config configs/c1000/base_v5_time.yaml

python scripts/evaluate.py \
    --config configs/c1000/base_v5_time.yaml \
    --checkpoint outputs/c1000/base_v5_time/best/model.safetensors \
    --splits train,valid,test
