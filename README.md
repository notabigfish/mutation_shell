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
    "wt_pos_pdb_number",
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
# full
python process_data.py --output_dir data/ --pdb_version pdb_260603 --pdb_format mmcif --re_group_seqadv --re_mutations --re_seqfasta --re_wholefasta --re_genmatchesm8 --re_gen_matching_dict --re_internalcsv --re_mutseqsv2 --re_cluster --num_workers 30 2>&1 | tee data.log

# only gen subset
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

## Training & Evaluation

### c1000
```bash
python scripts/train.py --config configs/c1000/base_v5.yaml
python scripts/evaluate.py --config configs/c1000/base_v5.yaml --checkpoint outputs/c1000/base_v5/checkpoint-13291/model.safetensors --splits test
```

# [Optional] Cluster Eval
```bash
python scripts/cluster_eval.py \
    --pred base_v3=outputs/c1000/base_v3/predictions_test.csv \
    --pred base_v5=outputs/c1000/base_v5/predictions_test.csv \
    --reference base_v3 \
    --candidate base_v5 \
    --out-dir outputs/c1000/base_v5_vs_base_v3
```


### call
```bash
# main model: base_v5
python scripts/build_alignment_sensitivity_data.py --base-config configs/call/base_v5.yaml --variant kabsch_all --subset call --out-config configs/call/base_v5_kabsch_all_gen.yaml --workers 16
# outputs:
#   configs/call/base_v5_kabsch_all_gen.yaml
#   data/alignment_sensitivity/call/kabsch_all
#   outputs/call/alignment_sensitivity

python scripts/train.py --config configs/call/base_v5.yaml 
python scripts/evaluate.py --config configs/call/base_v5.yaml  --checkpoint outputs/call/base_v5/best/model.safetensors --splits test

# evaluate other checkpoints
python scripts/evaluate.py \
  --config configs/call/base_v5.yaml \
  --checkpoint outputs/call/base_v5/checkpoint-10000/model.safetensors \
  --splits test \
  --output-dir outputs/call/base_v5/checkpoint-10000
```

## Experiment 2: Strict Baselines

### c1000

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
### call

```bash
python scripts/evaluate_baseline.py --config configs/call/zero_response.yaml
python scripts/evaluate_baseline.py --config configs/call/global_mean.yaml
python scripts/evaluate_baseline.py --config configs/call/shell_mean.yaml
python scripts/evaluate_baseline.py --config configs/call/mutation_type_shell_mean.yaml
```
```bash
python scripts/train.py --config configs/call/esm_mlp.yaml
python scripts/train.py --config configs/call/geometry_gnn.yaml
python scripts/train.py --config configs/call/coordinate_residual.yaml
```
```bash
python scripts/evaluate.py --config configs/call/base_v5_seed42.yaml --checkpoint outputs/call/base_v5_seed42/best/model.safetensors
python scripts/evaluate.py --config configs/call/base_v5_seed101.yaml --checkpoint outputs/call/base_v5_seed101/best/model.safetensors
python scripts/evaluate.py --config configs/call/base_v5_seed668.yaml --checkpoint outputs/call/base_v5_seed668/best/model.safetensors

python scripts/evaluate.py --config configs/call/esm_mlp.yaml --checkpoint outputs/call/esm_mlp/best/model.safetensors
python scripts/evaluate.py --config configs/call/geometry_gnn.yaml --checkpoint outputs/call/geometry_gnn/best/model.safetensors
python scripts/evaluate.py --config configs/call/coordinate_residual.yaml --checkpoint outputs/call/coordinate_residual/best/model.safetensors
```

```bash
for seed in 42 101 668
do
  python scripts/evaluate_strict_baselines.py \
      --pred base_v5_seed${seed}=outputs/call/base_v5_seed${seed}/predictions_test.csv \
      --pred zero_response=outputs/call/zero_response/predictions_test.csv \
      --pred global_mean=outputs/call/global_mean/predictions_test.csv \
      --pred shell_mean=outputs/call/shell_mean/predictions_test.csv \
      --pred mutation_type_shell_mean=outputs/call/mutation_type_shell_mean/predictions_test.csv \
      --pred esm_mlp=outputs/call/esm_mlp/predictions_test.csv \
      --pred geometry_gnn=outputs/call/geometry_gnn/predictions_test.csv \
      --pred coordinate_residual=outputs/call/coordinate_residual/predictions_test.csv \
      --candidate base_v5_seed${seed} \
      --n-bootstrap 10000 \
      --out-dir outputs/call/strict_baseline_comparison_base_v5_seed${seed}
done
```

## Additional Experiment: Threshold sensitivity

### c1000
```bash
python scripts/threshold_sensitivity.py \
  --pred base_v5=outputs/c1000/base_v5/predictions_test.csv \
  --pred zero_response=outputs/c1000/zero_response/predictions_test.csv \
  --pred global_mean=outputs/c1000/global_mean/predictions_test.csv \
  --pred shell_mean=outputs/c1000/shell_mean/predictions_test.csv \
  --pred mutation_type_shell_mean=outputs/c1000/mutation_type_shell_mean/predictions_test.csv \
  --pred esm_mlp=outputs/c1000/esm_mlp/predictions_test.csv \
  --pred geometry_gnn=outputs/c1000/geometry_gnn/predictions_test.csv \
  --pred coordinate_residual=outputs/c1000/coordinate_residual/predictions_test.csv \
  --response-thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 \
  --radius-thresholds 6 8 10 12 \
  --displacement-thresholds 0.5 1.0 1.5 2.0 \
  --out outputs/c1000/threshold_sensitivity.csv
```

### call
```bash
python scripts/threshold_sensitivity.py \
  --pred base_v5_seed42=outputs/call/base_v5_seed42/predictions_test.csv \
  --pred base_v5_seed101=outputs/call/base_v5_seed101/predictions_test.csv \
  --pred base_v5_seed668=outputs/call/base_v5_seed668/predictions_test.csv \
  --pred zero_response=outputs/call/zero_response/predictions_test.csv \
  --pred global_mean=outputs/call/global_mean/predictions_test.csv \
  --pred shell_mean=outputs/call/shell_mean/predictions_test.csv \
  --pred mutation_type_shell_mean=outputs/call/mutation_type_shell_mean/predictions_test.csv \
  --pred esm_mlp=outputs/call/esm_mlp/predictions_test.csv \
  --pred geometry_gnn=outputs/call/geometry_gnn/predictions_test.csv \
  --pred coordinate_residual=outputs/call/coordinate_residual/predictions_test.csv \
  --response-thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 \
  --radius-thresholds 6 8 10 12 \
  --displacement-thresholds 0.5 1.0 1.5 2.0 \
  --out outputs/call/threshold_sensitivity.csv
```

## Experiment 5: Counterfactual Test
### c1000
```bash
python scripts/counterfactual_tests.py \
  --config configs/c1000/base_v5.yaml \
  --checkpoint outputs/c1000/base_v5/best/model.safetensors \
  --split test \
  --out-dir outputs/c1000/base_v5_counterfactual 
```

### call
```bash
for seed in 42 101 668
do
  python scripts/counterfactual_tests.py \
    --config configs/call/base_v5_seed${seed}.yaml \
    --checkpoint outputs/call/base_v5_seed${seed}/best/model.safetensors \
    --split test \
    --response-threshold 0.1 \
    --displacement-threshold 1.0 \
    --radius-threshold 8.0 \
    --mut-aa-esm-mode negate_delta \
    --out-dir outputs/call/base_v5_seed${seed}/counterfactual
done
```

## Experiment 6: Alignment sensitivity

### c1000
Build one alignment-specific label set:

```bash
python scripts/build_alignment_sensitivity_data.py \
  --base-config configs/c1000/base_v5.yaml \
  --variant kabsch_exclude_4A \
  --out-config configs/c1000/base_v5_align_kabsch_exclude_4A.yaml
```

Build all variants:

```bash
TMALIGN_BIN="$(which TMalign)" \
bash scripts/run_alignment_sensitivity_build.sh
```

Train the four alignment-specific runs:

```bash
bash scripts/run_alignment_sensitivity.sh
```

Compare label sets before training:

```bash
python scripts/compare_alignment_labels.py \
  --reference data/alignment_sensitivity/c1000/kabsch_exclude_4A/samples_manifest.json \
  --candidate data/alignment_sensitivity/c1000/kabsch_all/samples_manifest.json \
  --out-dir outputs/alignment_sensitivity/label_compare_k4_vs_all \
  --num-workers 16
```

Evaluate four alignment-specific runs:
```bash
python scripts/evaluate.py --config configs/c1000/base_v5_align_kabsch_exclude_4A.yaml --checkpoint outputs/c1000/base_v5_align_kabsch_exclude_4A/best/model.safetensors
python scripts/evaluate.py --config configs/c1000/base_v5_align_kabsch_exclude_8A.yaml --checkpoint outputs/c1000/base_v5_align_kabsch_exclude_8A/best/model.safetensors
python scripts/evaluate.py --config configs/c1000/base_v5_align_kabsch_all.yaml --checkpoint outputs/c1000/base_v5_align_kabsch_all/best/model.safetensors
python scripts/evaluate.py --config configs/c1000/base_v5_align_tmalign.yaml --checkpoint outputs/c1000/base_v5_align_tmalign/best/model.safetensors
```

Collect run outputs:

```bash
python scripts/collect_alignment_sensitivity_results.py \
  --run kabsch_exclude_4A=outputs/c1000/base_v5_align_kabsch_exclude_4A \
  --run kabsch_exclude_8A=outputs/c1000/base_v5_align_kabsch_exclude_8A \
  --run kabsch_all=outputs/c1000/base_v5_align_kabsch_all \
  --run tmalign=outputs/c1000/base_v5_align_tmalign \
  --out-dir outputs/alignment_sensitivity/final
```

Cluster-level paired comparison:

```bash
python scripts/cluster_compare_alignment_sensitivity.py \
  --pred kabsch_exclude_4A=outputs/c1000/base_v5_align_kabsch_exclude_4A/predictions_test.csv \
  --pred kabsch_exclude_8A=outputs/c1000/base_v5_align_kabsch_exclude_8A/predictions_test.csv \
  --pred kabsch_all=outputs/c1000/base_v5_align_kabsch_all/predictions_test.csv \
  --pred tmalign=outputs/c1000/base_v5_align_tmalign/predictions_test.csv \
  --reference kabsch_exclude_4A \
  --out-dir outputs/alignment_sensitivity/cluster_compare
```


### call
Build alignment-specific label sets:

```bash
for variant in kabsch_all kabsch_exclude_4A kabsch_exclude_8A
do
  python scripts/build_alignment_sensitivity_data.py \
    --base-config configs/call/base_v5_seed$42.yaml \
    --variant ${variant} \
    --out-config configs/call/base_v5_${variant}.yaml \
    --subset call
done
```

Train the 3 alignment-specific runs:

```bash
for variant in kabsch_all kabsch_exclude_4A kabsch_exclude_8A
do
  python scripts/train.py --config configs/call/base_v5_${vairant}.yaml
done
```

Compare label sets before training:

```bash
for ref_name in  kabsch_exclude_4A kabsch_exclude_8A
do
  python scripts/compare_alignment_labels.py \
    --reference data/alignment_sensitivity/call/${ref_name}/samples_manifest.json \
    --candidate data/alignment_sensitivity/call/kabsch_all/samples_manifest.json \
    --out-dir outputs/call/alignment_sensitivity/label_compare_${ref_name}_vs_all \
    --num-workers 8
done
```

Evaluate 3 alignment-specific runs:
```bash
for ref_name in kabsch_all kabsch_exclude_4A kabsch_exclude_8A
do
  python scripts/evaluate.py --config configs/call/base_v5_${ref_name}.yaml --checkpoint outputs/call/base_v5_${ref_name}/best/model.safetensors
done
```

Collect run outputs:

```bash
python scripts/collect_alignment_sensitivity_results.py \
  --run kabsch_exclude_4A=outputs/call/base_v5_kabsch_exclude_4A \
  --run kabsch_exclude_8A=outputs/call/base_v5_kabsch_exclude_8A \
  --run kabsch_all=outputs/call/base_v5_kabsch_all \
  --run base_v5_seed42=outputs/call/base_v5_seed42 \
  --run base_v5_seed101=outputs/call/base_v5_seed101 \
  --run base_v5_seed668=outputs/call/base_v5_seed668 \
  --out-dir outputs/call/alignment_sensitivity/final
```

Cluster-level paired comparison:

```bash
for ref_name in kabsch_exclude_4A kabsch_exclude_8A kabsch_all base_v5_seed42 base_v5_seed101 base_v5_seed668
do
  python scripts/cluster_compare_alignment_sensitivity.py \
    --pred kabsch_exclude_4A=outputs/call/base_v5_kabsch_exclude_4A/predictions_test.csv \
    --pred kabsch_exclude_8A=outputs/call/base_v5_kabsch_exclude_8A/predictions_test.csv \
    --pred kabsch_all=outputs/call/base_v5_kabsch_all/predictions_test.csv \
    --pred base_v5_seed42=outputs/call/base_v5_seed42/predictions_test.csv \
    --pred base_v5_seed101=outputs/call/base_v5_seed101/predictions_test.csv \
    --pred base_v5_seed668=outputs/call/base_v5_seed668/predictions_test.csv \
    --reference ${ref_name} \
    --out-dir outputs/call/alignment_sensitivity/cluster_compare_ref_${ref_name}
done
```

## Experiment 7: Biological stratification

### c1000

Fetch domain annotations:
```bash
python scripts/fetch_domain_annotations.py \
  --sample-csv data/SingleMutPairs2024_subset_c1000.csv \
  --out-csv data/domain_annotations_subset_c1000.csv \
  --cache-json data/cache/pdbe_sifts_domain_cache_subset_c1000.json \
  --sources CATH,Pfam,SCOP,InterPro
```

Single-model biological stratification:

```bash
python scripts/evaluate_biological_stratification.py \
  --config configs/c1000/base_v5.yaml \
  --pred MuSRNet=outputs/c1000/base_v5/predictions_test.csv \
  --sample-csv data/SingleMutPairs2024_subset_c1000.with_wt_pos.csv \
  --pdb-dir data/pdb \
  --domain-annotations data/domain_annotations_subset_c1000.csv \
  --out-dir outputs/c1000/base_v5/biological_stratification/ \
  --num-workers 30 2>&1 | tee out.log
```

Reference-vs-candidate stratified comparison:

```bash
python scripts/evaluate_biological_stratification.py \
  --config configs/c1000/base_v5.yaml \
  --pred base_v5=outputs/c1000/base_v5/predictions_test.csv \
  --pred shell_mean=outputs/c1000/shell_mean/predictions_test.csv \
  --reference shell_mean \
  --candidate base_v5 \
  --sample-csv data/SingleMutPairs2024_subset_c1000.with_wt_pos.csv \
  --pdb-dir data/pdb \
  --domain-annotations data/domain_annotations_subset_c1000.csv \
  --out-dir outputs/c1000/base_v5/vs_shell_mean \
  --num-workers 30
```


### call
Fetch domain annotations:
```bash
python scripts/fetch_domain_annotations.py \
  --sample-csv data/SingleMutPairs2024.csv \
  --out-csv data/domain_annotations.csv \
  --cache-json data/cache/pdbe_sifts_domain_cache.json \
  --sources CATH,Pfam,SCOP,InterPro
```

Download DSSP:
```bash
python scripts/download_dssp.py
```

```bash
for seed in 42 101 668
do
  python scripts/evaluate_biological_stratification.py \
    --config configs/call/base_v5_seed${seed}.yaml \
    --pred MuSRNet=outputs/call/base_v5_seed${seed}/predictions_test.csv \
    --sample-csv data/SingleMutPairs2024.csv \
    --pdb-dir data/pdb \
    --domain-annotations data/domain_annotations.csv \
    --out-dir outputs/call/base_v5_seed${seed}/biological_stratification/ \
    --num-workers 16 2>&1 | tee out.log
done
```

Reference-vs-candidate stratified comparison:

```bash
for ref in zero_response global_mean shell_mean mutation_type_shell_mean esm_mlp geometry_gnn coordinate_residual
  do
    for seed in 42 101 668
    do
      python scripts/evaluate_biological_stratification.py \
        --config configs/call/base_v5_seed${seed}.yaml \
        --pred base_v5_seed${seed}=outputs/call/base_v5_seed${seed}/predictions_test.csv \
        --pred ${ref}=outputs/call/${ref}/predictions_test.csv \
        --reference ${ref} \
        --candidate base_v5_seed${seed} \
        --sample-csv data/SingleMutPairs2024.csv \
        --pdb-dir data/pdb \
        --domain-annotations data/domain_annotations.csv \
        --out-dir outputs/call/biological_stratification/v5_seed${seed}_vs_${ref} \
        --num-workers 16
    done
  done
```


## Experiment 8: Time-split generalization

### c1000
Create the cluster-safe time split:

```bash
python scripts/create_time_split.py \
  --config configs/c1000/base_v5.yaml \
  --csv data/SingleMutPairs2024_subset_c1000.csv \
  --out data/processed/splits_time_c1000_2023_2024_2025.json \
  --cluster-policy latest_release \
  --train-max-year 2023 \
  --valid-year 2024 \
  --test-min-year 2025
```

Train and evaluate `base_v5` on the time split:

```bash
python scripts/train.py --config configs/c1000/base_v5_time.yaml
python scripts/evaluate.py \
  --config configs/c1000/base_v5_time.yaml \
  --checkpoint outputs/c1000/base_v5_time/best/model.safetensors \
  --splits train,valid,test
```

Write the compact time-split report:

```bash
python scripts/report_time_split.py \
  --config configs/c1000/base_v5_time.yaml \
  --audit-dir data/processed/time_split_c1000_2023_2024_2025_audit \
  --output-dir outputs/c1000/base_v5_time
```


### call
Create the cluster-safe time split:

```bash
python scripts/create_time_split.py \
  --config configs/call/base_v5_seed42.yaml \
  --csv data/SingleMutPairs2024.csv \
  --out data/processed/splits_time_call_2023_2024_2025.json \
  --cluster-policy latest_release \
  --train-max-year 2023 \
  --valid-year 2024 \
  --test-min-year 2025
```

Train and evaluate `base_v5` on the time split:

```bash
python scripts/train.py --config configs/call/base_v5_time.yaml
python scripts/evaluate.py \
  --config configs/call/base_v5_time.yaml \
  --checkpoint outputs/call/base_v5_time/best/model.safetensors \
  --splits train,valid,test
```

Write the compact time-split report:

```bash
python scripts/report_time_split.py \
  --config configs/call/base_v5_time.yaml \
  --audit-dir data/processed/time_split_call_2023_2024_2025_audit \
  --output-dir outputs/call/base_v5_time
```



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
