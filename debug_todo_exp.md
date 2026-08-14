三、现有评估代码必须先修正
1. 统一 cluster metrics

```bash
[running]for seed in 42 101 668
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

四、必须补的三个机制实验

先运行 no-context，以决定最终是否保留该模块：
```bash
[running] python scripts/train.py --config configs/call/base_v5_no_context_seed42.yaml
python scripts/evaluate.py --config configs/call/base_v5_no_context_seed42.yaml \
  --checkpoint outputs/call/base_v5_no_context_seed42/best/model.safetensors --splits test
```

然后根据no-context结果修改：
```bash
python - <<'PY'
import copy, yaml
from pathlib import Path
src = yaml.safe_load(Path("configs/call/base_v5_seed42.yaml").read_text())
use_context = True  # 根据 no-context 结果修改

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

for name in base_v5_pure_hurdle_seed42 base_v5_direct_seed42 base_v5_global_loss_seed42
do
  python scripts/train.py --config configs/call/${name}.yaml
  python scripts/evaluate.py --config configs/call/${name}.yaml \
    --checkpoint outputs/call/${name}/best/model.safetensors --splits test
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

五、修正反事实实验

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


七、Threshold sensitivity：只在 validation 选阈值

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

八、Time split 必须重做

1. 必须补：同一 Time split 下的 baseline

`zero_response/global_mean/esm_mlp/coordinate_residual` **Time split 不必重跑**。前两个太弱，后两个要额外训练，信息增益不够。Time split 是 robustness experiment，不需要把 Experiment 2 整套重新复制。


```bash
python scripts/train.py --config configs/call/geometry_gnn_time.yaml
python scripts/evaluate.py --config configs/call/geometry_gnn_time.yaml --checkpoint outputs/call/geometry_gnn_time/best/model.safetensors --splits valid,test
```

```bash
for m in shell_mean mutation_type_shell_mean
do
  python scripts/evaluate_baseline.py --config configs/call/${m}_time.yaml --splits valid,test
done
```

---

2. 必须补：Time-test 的 cluster-level paired statistics


```bash
python scripts/evaluate_strict_baselines.py \
  --pred base_v5_time=outputs/call/base_v5_time/predictions_test.csv \
  --pred geometry_gnn_time=outputs/call/geometry_gnn_time/predictions_test.csv \
  --pred shell_mean_time=outputs/call/shell_mean_time/predictions_test.csv \
  --pred mutation_type_shell_mean_time=outputs/call/mutation_type_shell_mean_time/predictions_test.csv \
  --candidate base_v5_time \
  --n-bootstrap 10000 \
  --out-dir outputs/call/time_split_comparison
```

最终重点看：

```text
outputs/call/time_split_comparison/summary_metrics.csv
outputs/call/time_split_comparison/statistical_tests.csv
```

Time-split 论文结论标准

如果要写：

> MuSRNet's advantage persists under temporal distribution shift.

至少要求：

```text
test Shell-MAE < geometry_gnn_time
test AUPRC > geometry_gnn_time
test Shell-MAE < shell_mean_time / mutation_type_shell_mean_time
test AUPRC > shell_mean_time / mutation_type_shell_mean_time
```

更强版本：

```text
geometry_gnn_time 的 AUPRC paired bootstrap CI > 0
Shell-MAE paired CI 最好 < 0；如果跨 0，则写 comparable displacement accuracy，不要写 significant improvement
```


3. 必须补：label-distribution shift audit

这是现在 Time split 最大的解释缺口之一。

AUPRC **受 positive prevalence 强烈影响**。当前 random test AUPRC 是 `0.159`，time-test 却是 `0.224`，不能直接说 temporal generalization 上 AUPRC 反而更好；可能只是 later split 中 perturbed residue 比例变高。

所以至少报告：

```text
valid/test perturbed fraction
valid/test mean true displacement
AUPRC / perturbed fraction
```

直接重新生成 report：

```bash
python scripts/report_time_split.py \
  --config configs/call/base_v5_time.yaml \
  --audit-dir data/processed/time_split_call_2023_2024_2025_audit \
  --output-dir outputs/call/base_v5_time
```

4. `latest_release` 要不要改？

**如果你确定保留当前设计，不改。**

但当前 policy 是：

```text
cluster_policy = latest_release
```

所以 test cluster 可以包含更早年份的 sample。此时论文最好叫：

> **later-release-associated cluster temporal split**

或者：

> **cluster-safe temporal stress test**

而不是非常严格的：

> every test structure was released after training cutoff.

这与“2023/2024/2025 选得是否合理”是两个问题。

如果你接受前一种表述，`create_time_split.py` **完全不用改**。



九、Biological stratification：不需要重算所有模型

```bash
for ref in geometry_gnn coordinate_residual
do
  python scripts/evaluate_biological_stratification.py --config configs/call/base_v5_seed42.yaml --pred ${ref}=outputs/call/${ref}/predictions_test.csv --pred base_v5_seed42=outputs/call/base_v5_seed42/predictions_test.csv --reference ${ref} --candidate base_v5_seed42 --sample-csv data/SingleMutPairs2024.with_wt_pos.csv --pdb-dir data/pdb --domain-annotations data/domain_annotations.csv --out-dir outputs/call/biological_stratification/v5_seed42_vs_${ref} --num-workers 30
done
```

跑完看：
```bash
python - <<'PY'
import pandas as pd
p="outputs/call/biological_stratification/v5_seed42_vs_geometry_gnn/pairwise_stratified_diff.csv"
d=pd.read_csv(p)
print(d[d.claimable][["factor","group","metric","mean_diff","bootstrap_ci_low","bootstrap_ci_high","wilcoxon_p","q_bh"]].to_string(index=False))
PY

python - <<'PY'
import pandas as pd
p="outputs/call/biological_stratification/v5_seed42_vs_coordinate_residual/pairwise_stratified_diff.csv"
d=pd.read_csv(p)
print(d[d.claimable][["factor","group","metric","mean_diff","bootstrap_ci_low","bootstrap_ci_high","wilcoxon_p","q_bh"]].to_string(index=False))
PY
```