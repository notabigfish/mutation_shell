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
python process_data.py --output_dir data/ --pdb_version pdb_260603 --pdb_format mmcif --re_subset
```

```bash
python scripts/prepare_data.py --csv data/SingleMutPairs2024.csv  --out data/processed/samples.pt
```

This creates `data/processed/samples.pt` as a manifest and stores one processed sample dictionary per file under `data/processed/samples/`.

## ESM precomputation

```bash
python scripts/precompute_esm.py --samples data/processed/samples.pt --out-esm-lmdb data/processed/esm.lmdb --out-filtered-manifest data/processed/samples.pt
```

## Training & Evaluation
```bash
# main model: base_v5
python scripts/build_alignment_sensitivity_data.py --base-config configs/call/base_v5.yaml --variant kabsch_all --subset call --out-config configs/call/base_v5_kabsch_all_gen.yaml --workers 16
# outputs:
#   configs/call/base_v5_kabsch_all_gen.yaml
#   data/alignment_sensitivity/call/kabsch_all
#   outputs/call/alignment_sensitivity

for seed in 42 101 668
do
  python scripts/train.py --config configs/call/base_v5_seed${seed}.yaml
  python scripts/evaluate.py \
    --config configs/call/base_v5_seed${seed}.yaml \
    --checkpoint outputs/call/base_v5_seed${seed}/best/model.safetensors \
    --splits test
done
```

## Experiment 2: Strict Baselines
```bash
for taskname in zero_response global_mean shell_mean mutation_type_shell_mean
do 
  python scripts/evaluate_baseline.py --config configs/call/${taskname}.yaml
done

for taskname in esm_mlp geometry_gnn coordinate_residual
do
  python scripts/train.py --config configs/call/${taskname}.yaml
  python scripts/evaluate.py --config configs/call/${taskname}.yaml --checkpoint outputs/call/${taskname}/best/model.safetensors
 
done


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
      --out-dir outputs/call/strict_baseline_comparison_base_v5_seed${seed}
done
```

## Additional Experiment: Threshold sensitivity

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

修正版：
```bash
python scripts/evaluate.py --config configs/call/base_v5_seed42.yaml \
  --checkpoint outputs/call/base_v5_seed42/best/model.safetensors --splits valid,test

python scripts/threshold_sensitivity.py \
  --pred base_v5_seed42=outputs/call/base_v5_seed42/predictions_valid.csv \
  --response-thresholds 0.02 0.05 0.1 0.2 0.3 0.5 \
  --radius-thresholds 8 \
  --displacement-thresholds 1 \
  --out outputs/call/threshold_sensitivity_valid.csv
```

## Experiment 5: Counterfactual Test

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

修正版：
```bash
for s in 0 1 2 3 4 5 6 7 8 9
do
  python scripts/counterfactual_tests.py \
    --config configs/call/base_v5_seed42.yaml \
    --checkpoint outputs/call/base_v5_seed42/best/model.safetensors \
    --split test \
    --seed ${s} \
    --one-per-cluster \
    --response-threshold 0.1 \
    --displacement-threshold 1.0 \
    --radius-threshold 8.0 \
    --out-dir outputs/call/base_v5_seed42/counterfactual_corrected_seed${s}
done
```

## Experiment 6: Alignment sensitivity
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
for ref in geometry_gnn coordinate_residual
do
  python scripts/evaluate_biological_stratification.py \
    --config configs/call/base_v5_seed42.yaml \
    --pred ${ref}=outputs/call/${ref}/predictions_test.csv \
    --pred base_v5_seed42=outputs/call/base_v5_seed42/predictions_test.csv \
    --reference ${ref} \
    --candidate base_v5_seed42 \
    --sample-csv data/SingleMutPairs2024.csv \
    --pdb-dir data/pdb \
    --domain-annotations data/domain_annotations.csv \
    --out-dir outputs/call/biological_stratification/v5_seed42_vs_${ref} \
    --num-workers 16
done
```

跑完看：
```bash
python - <<'PY'
import pandas as pd
p="outputs/call/biological_stratification/v5_seed42_vs_geometry_gnn/pairwise_stratified_diff.csv"
d=pd.read_csv(p)
with open("outputs/call/v5_seed42_vs_geometry_gnn/show_results.txt","w") as f:
  print(d[d.claimable][["factor","group","metric","mean_diff","bootstrap_ci_low","bootstrap_ci_high","wilcoxon_p","q_bh"]].to_string(index=False,float_format=lambda x: f"{x:.3f}"), file=f)
PY
```

## Experiment 8: Time-split generalization
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
for taskname in base_v5 geometry_gnn
do
  python scripts/train.py --config configs/call/${taskname}_time.yaml
  python scripts/evaluate.py \
    --config configs/call/${taskname}_time.yaml \
    --checkpoint outputs/call/${taskname}_time/best/model.safetensors \
    --splits valid,test
done

for m in shell_mean mutation_type_shell_mean
do
  python scripts/evaluate_baseline.py --config configs/call/${m}_time.yaml --splits valid,test
done
```

Time-test cluster-level paired statistics:
```
python scripts/evaluate_strict_baselines.py \
  --pred base_v5_time=outputs/call/base_v5_time/predictions_test.csv \
  --pred geometry_gnn_time=outputs/call/geometry_gnn_time/predictions_test.csv \
  --pred shell_mean_time=outputs/call/shell_mean_time/predictions_test.csv \
  --pred mutation_type_shell_mean_time=outputs/call/mutation_type_shell_mean_time/predictions_test.csv \
  --candidate base_v5_time \
  --n-bootstrap 10000 \
  --out-dir outputs/call/time_split_comparison
```

Write the compact time-split report:

```bash
python scripts/report_time_split.py \
  --config configs/call/base_v5_time.yaml \
  --audit-dir data/processed/time_split_call_2023_2024_2025_audit \
  --output-dir outputs/call/base_v5_time
```

## Experiment 9: Mechanism Study
### No-context vs context
```bash
python scripts/train.py --config configs/call/base_v5_no_context_seed42.yaml
python scripts/evaluate.py \
  --config configs/call/base_v5_no_context_seed42.yaml \
  --checkpoint outputs/call/base_v5_no_context_seed42/checkpoint-419952/model.safetensors \
  --splits test, valid
```

Compare `base_v5_seed42` and `base_v5_no_context_seed42`:
```bash
python scripts/evaluate_strict_baselines.py \
  --pred context=outputs/call/base_v5_seed42/predictions_valid.csv \
  --pred no_context=outputs/call/base_v5_no_context_seed42/predictions_valid_419952.csv \
  --candidate no_context \
  --n-bootstrap 10000 \
  --out-dir outputs/call/context_ablation_valid/no_context_419952
```

然后只看：
```text
summary_metrics.csv:
shell_mae
perturbed_auprc
cluster_avg_shell_mae
cluster_avg_auprc

statistical_tests.csv:
shell_mae -> mean_diff, bootstrap_95ci_low/high
perturbed_auprc -> mean_diff, bootstrap_95ci_low/high
```

shell_mae: context 0.273 vs no_context 0.365，context 明显更好。
perturbed_auprc: 0.189 vs 0.134，context 明显更好。
AUROC: 0.790 vs 0.708，context 更好。
cluster_avg_auprc: 0.360 vs 0.346，context 更好。
cluster_avg_shell_mae: no_context 仅小幅更好 0.633 vs 0.637，但 paired CI [-0.038, 0.026]、p=0.796，没有统计证据支持 no-context 更好。

然后根据no-context结果修改：

```bash
python - <<'PY'
import copy, yaml
from pathlib import Path
src = yaml.safe_load(Path("configs/call/base_v5_seed42.yaml").read_text())
use_context = True  # no-context结果表明应该用True

runs = {
    "base_v5_pure_hurdle_seed42": {"response_mode": "pure_hurdle"},
    "base_v5_direct_seed42": {"response_mode": "direct"},
    "base_v5_global_loss_seed42": {"response_mode": "background_excess"},
}
for name, patch in runs.items():
    c = copy.deepcopy(src)
    c["paths"]["output_dir"] = f"outputs/call/{name}"
    c["wandb"]["run_name"] = name
    c["model"].update(patch)
    c["model"]["use_mutation_context"] = use_context
    if patch["response_mode"] != "background_excess":
        c["loss"]["w_background"] = 0.0
    if "global_loss" in name:
        c["loss"]["reduction"] = "global"
    Path(f"configs/call/{name}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
PY

for taskname in pure_hurdle direct global_loss
do
  python scripts/train.py --config configs/call/base_v5_${taskname}_seed42.yaml
  python scripts/evaluate.py \
    --config configs/call/base_v5_${taskname}_seed42.yaml \
    --checkpoint outputs/call/base_v5_${taskname}_seed42/best/model.safetensors \
    --splits test
done
```

然后比较：
```bash
python scripts/evaluate_strict_baselines.py \
  --pred full=outputs/call/base_v5_seed42/predictions_test.csv \
  --pred pure_hurdle=outputs/call/base_v5_pure_hurdle_seed42/predictions_test.csv \
  --pred direct=outputs/call/base_v5_direct_seed42/predictions_test.csv \
  --pred global_loss=outputs/call/base_v5_global_loss_seed42/predictions_test.csv \
  --candidate full \
  --n-bootstrap 10000 \
  --out-dir outputs/call/mechanism_closure_seed42
```