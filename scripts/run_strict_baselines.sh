#!/usr/bin/env bash
set -e

python scripts/evaluate_baseline.py --config configs/c1000/zero_response.yaml
python scripts/evaluate_baseline.py --config configs/c1000/global_mean.yaml
python scripts/evaluate_baseline.py --config configs/c1000/shell_mean.yaml
python scripts/evaluate_baseline.py --config configs/c1000/mutation_type_shell_mean.yaml

python scripts/train.py --config configs/c1000/esm_mlp.yaml
python scripts/train.py --config configs/c1000/geometry_gnn.yaml
python scripts/train.py --config configs/c1000/coordinate_residual.yaml

# Adjust checkpoint paths if your Trainer output structure differs.
python scripts/evaluate.py --config configs/c1000/base_v5.yaml --checkpoint outputs/c1000/base_v5/best/model.safetensors
python scripts/evaluate.py --config configs/c1000/esm_mlp.yaml --checkpoint outputs/c1000/esm_mlp/best/model.safetensors
python scripts/evaluate.py --config configs/c1000/geometry_gnn.yaml --checkpoint outputs/c1000/geometry_gnn/best/model.safetensors
python scripts/evaluate.py --config configs/c1000/coordinate_residual.yaml --checkpoint outputs/c1000/coordinate_residual/best/model.safetensors

python scripts/evaluate_strict_baselines.py \
    --pred base_v5=outputs/c1000/base_v5/predictions_test.csv \
    --pred zero_response=outputs/c1000/zero_response/predictions_test.csv \
    --pred global_mean=outputs/c1000/global_mean/predictions_test.csv \
    --pred shell_mean=outputs/c1000/shell_mean/predictions_test.csv \
    --pred mutation_type_shell_mean=outputs/c1000/mutation_type_shell_mean/predictions_test.csv \
    --pred esm_mlp=outputs/c1000/esm_mlp/predictions_test.csv \
    --pred geometry_gnn=outputs/c1000/geometry_gnn/predictions_test.csv \
    --pred coordinate_residual=outputs/c1000/coordinate_residual/predictions_test.csv \
    --candidate base_v5 \
    --out-dir outputs/c1000/strict_baseline_comparison
