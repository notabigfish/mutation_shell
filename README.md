# MuSRNet
## Input data

The main CSV file is `data/SingleMutPairs2024.csv` with columns:

```python
[
    "sample_id",
    "wt_pdb_id",
    "wt_chain_id",
    "mut_pdb_id",
    "mut_chain_id",
    "mut_pos_seq_index",
    "mut_pos_pdb_number",
    "wt_aa_type",
    "mut_aa_type",
    "wt_sequence",
    "mut_sequence",
    "cluster_id_30",
    "release_date",
]
```

PDB/mmCIF files must be available under `data/pdb/` as lowercase `{pdb_id}.cif.gz`.

## Env

```bash
conda activate pt311cu130
cd /rds/projects/l/liuje-multiai/shuo/mutation/MuSRNet
```

## Preprocessing

```bash
python process_data.py --output_dir data/ --pdb_version pdb_260603 --pdb_format mmcif --re_subset --n_clusters 1000
```

```bash
python scripts/prepare_data.py --csv data/SingleMutPairs2024_subset_c1000.csv  --out data/processed/samples_subset_c1000_raw.pt
```

This creates `data/processed/samples.pt` as a manifest and stores one processed sample dictionary per file under `data/processed/samples/`.

## ESM precomputation

```bash
python scripts/precompute_esm.py --samples data/processed/samples_subset_c1000_raw.pt --out-esm-lmdb data/processed/esm_subset_c1000.lmdb --out-filtered-manifest data/processed/samples_subset_c1000.pt
```

## Training

```bash
python scripts/train.py --config configs/c1000/base_v1.yaml
python scripts/train.py --config configs/c1000/base_v2.yaml
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/c1000/base_v5.yaml --checkpoint outputs/c1000/base_v5/checkpoint-13291/model.safetensors
```

### Cluster Eval
```bash
python scripts/cluster_eval.py \
    --pred base_v3=outputs/c1000/base_v3/predictions_test.csv \
    --pred base_v5=outputs/c1000/base_v5/predictions_test.csv \
    --reference base_v3 \
    --candidate base_v5 \
    --out-dir outputs/c1000/base_v5_vs_base_v3
```
## Strict Baselines

```bash
python scripts/evaluate_baseline.py --config configs/c1000/zero_response.yaml
python scripts/evaluate_baseline.py --config configs/c1000/global_mean.yaml
python scripts/evaluate_baseline.py --config configs/c1000/shell_mean.yaml
python scripts/evaluate_baseline.py --config configs/c1000/mutation_type_shell_mean.yaml
```
```bash
python scripts/train.py --config configs/c1000/esm_mlp.yaml
python scripts/train.py --config configs/c1000/geometry_gnn.yaml
python scripts/train.py --config configs/c1000/coordinate_residual.yaml
```
```bash
python scripts/evaluate.py --config configs/c1000/base_v5.yaml --checkpoint outputs/c1000/base_v5/best/model.safetensors
python scripts/evaluate.py --config configs/c1000/esm_mlp.yaml --checkpoint outputs/c1000/esm_mlp/best/model.safetensors
python scripts/evaluate.py --config configs/c1000/geometry_gnn.yaml --checkpoint outputs/c1000/geometry_gnn/best/model.safetensors
python scripts/evaluate.py --config configs/c1000/coordinate_residual.yaml --checkpoint outputs/c1000/coordinate_residual/best/model.safetensors
```

```bash
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
```

## Outputs

- `data/processed/samples.pt`
- `data/processed/samples/*.pt`
- `data/processed/esm/*.pt`
- `data/processed/splits.json`
- `outputs/best.pt`
- `outputs/last.pt`
- `outputs/train_log.csv`
- `outputs/config_used.yaml`
- `outputs/eval_train.json`
- `outputs/eval_valid.json`
- `outputs/eval_test.json`
- `outputs/predictions_test.csv`




## Backup
### sanity check | graph direction
```bash
python scripts/sanity_check_graph_direction.py --config configs/c1000/base_v2.yaml --num-samples 20
```
Should output `ALL PASSED.` If not -> wrong edge_index direction or kNN dst!=center

### sanity check | 32-sample overfit
```bash
python scripts/sanity_check_overfit_32.py --config configs/c1000/base_v2.yaml --num-samples 32 --steps 1500 --batch-size 4 --lr 3e-4
```
Should output `PASSED: model can overfit 32 samples`. 

### sanity check | node-label
```bash
python scripts/sanity_check_labels.py --config configs/c1000/base_v2.yaml --num-samples 100
```
Should output `PASSED: labels/features have consistent node lengths.`
