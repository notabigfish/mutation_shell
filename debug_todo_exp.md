三、现有评估代码必须先修正
1. 统一 cluster metrics

```bash
[done]for seed in 42 101 668
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
[done] python scripts/train.py --config configs/call/base_v5_no_context_seed42.yaml

[done] python scripts/evaluate.py \
  --config configs/call/base_v5_no_context_seed42.yaml \
  --checkpoint outputs/call/base_v5_no_context_seed42/best/model.safetensors \
  --splits test, valid
```

比较`base_v5_seed42` 和 `base_v5_no_context_seed42`
```bash
[done] python scripts/evaluate_strict_baselines.py \
  --pred context=outputs/call/base_v5_seed42/predictions_valid.csv \
  --pred no_context=outputs/call/base_v5_no_context_seed42/predictions_valid.csv \
  --candidate no_context \
  --n-bootstrap 10000 \
  --out-dir outputs/call/context_ablation_valid
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

| validation 结果                        | 设置                        |
| ------------------------------------ | ------------------------- |
| no-context `shell_mae ↓` 且 `AUPRC ↑` | `False`                   |
| context `shell_mae ↓` 且 `AUPRC ↑`    | `True`                    |
| no-context `shell_mae ↓`，AUPRC 基本持平  | `False`                   |
| no-context AUPRC ↑，shell-MAE 基本持平    | `False`                   |
| context `shell_mae ↓`，AUPRC 基本持平     | `True`                    |
| context AUPRC ↑，shell-MAE 基本持平       | `True`                    |
| 一个明显改善 Shell-MAE、另一个明显改善 AUPRC | 看 cluster-level paired CI |
| 两边都没有明确优势                            | `False`                   |

对于 Shell-MAE：
```text
diff = no_context - context
CI < 0     → no-context 明确更好
CI > 0     → context 明确更好
CI 跨 0    → 没有明确差异
```
对于 AUPRC：
```text
diff = no_context - context
CI > 0     → no-context 明确更好
CI < 0     → context 明确更好
CI 跨 0    → 没有明确差异
```

然后根据no-context结果修改：
```bash
[done] python - <<'PY'
import copy, yaml
from pathlib import Path
src = yaml.safe_load(Path("configs/call/base_v5_seed42.yaml").read_text())
use_context = False  # 根据 no-context 结果修改

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

[done] python scripts/train.py --config configs/call/base_v5_pure_hurdle_seed42.yaml
[done] python scripts/evaluate.py --config configs/call/base_v5_pure_hurdle_seed42.yaml --checkpoint outputs/call/base_v5_pure_hurdle_seed42/best/model.safetensors --splits test

[done] python scripts/train.py --config configs/call/base_v5_direct_seed42.yaml
[done] python scripts/evaluate.py --config configs/call/base_v5_direct_seed42.yaml --checkpoint outputs/call/base_v5_direct_seed42/best/model.safetensors --splits test

[done] python scripts/train.py --config configs/call/base_v5_global_loss_seed42.yaml
[done] python scripts/evaluate.py --config configs/call/base_v5_global_loss_seed42.yaml --checkpoint outputs/call/base_v5_global_loss_seed42/best/model.safetensors --splits test

```

然后比较：
```bash
[done] python scripts/evaluate_strict_baselines.py \
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
[done] for s in 0 1 2 3 4 5 6 7 8 9
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
[done] python scripts/evaluate.py --config configs/call/base_v5_seed42.yaml \
  --checkpoint outputs/call/base_v5_seed42/best/model.safetensors --splits valid,test

[done] python scripts/threshold_sensitivity.py \
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
[done] python scripts/train.py --config configs/call/geometry_gnn_time.yaml

[done] python scripts/evaluate.py --config configs/call/geometry_gnn_time.yaml --checkpoint outputs/call/geometry_gnn_time/best/model.safetensors --splits valid,test
```

```bash
[done] for m in shell_mean mutation_type_shell_mean
do
  python scripts/evaluate_baseline.py --config configs/call/${m}_time.yaml --splits valid,test
done
```

---

2. 必须补：Time-test 的 cluster-level paired statistics


```bash
[done] python scripts/evaluate_strict_baselines.py \
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
[done] python scripts/report_time_split.py \
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
[done] for ref in geometry_gnn coordinate_residual
do
  python scripts/evaluate_biological_stratification.py --config configs/call/base_v5_seed42.yaml --pred ${ref}=outputs/call/${ref}/predictions_test.csv --pred base_v5_seed42=outputs/call/base_v5_seed42/predictions_test.csv --reference ${ref} --candidate base_v5_seed42 --sample-csv data/SingleMutPairs2024.csv --pdb-dir data/pdb --domain-annotations data/domain_annotations.csv --out-dir outputs/call/biological_stratification/v5_seed42_vs_${ref} --num-workers 16
done
```

跑完看：
```bash
[done] python - <<'PY'
import pandas as pd
p="outputs/call/biological_stratification/v5_seed42_vs_geometry_gnn/pairwise_stratified_diff.csv"
d=pd.read_csv(p)
print(d[d.claimable][["factor","group","metric","mean_diff","bootstrap_ci_low","bootstrap_ci_high","wilcoxon_p","q_bh"]].to_string(index=False))
PY

outputs:
                   factor                        group           metric  mean_diff  bootstrap_ci_low  bootstrap_ci_high   wilcoxon_p         q_bh
           exposure_group                      exposed    class_correct   0.275193          0.184701           0.364356 3.046536e-07 2.766584e-06
           exposure_group                      exposed non_local_recall   0.487620          0.404048           0.571222 6.578193e-15 2.762841e-13
           exposure_group                       buried  perturbed_auprc   0.040719          0.020006           0.064167 1.477505e-03 1.079221e-02
           exposure_group                       buried    class_correct   0.340518          0.261853           0.422086 3.436282e-12 6.076793e-11
           exposure_group                       buried non_local_recall   0.486473          0.401386           0.568657 5.938609e-16 3.325621e-14
  substitution_size_group               large_to_small    class_correct   0.330751          0.248825           0.409723 1.060925e-11 1.782354e-10
  substitution_size_group               large_to_small non_local_recall   0.531663          0.452241           0.610901 1.667107e-19 1.400370e-17
  substitution_size_group               small_to_large  perturbed_auprc   0.033639          0.013288           0.053260 3.804966e-04 3.043973e-03
  substitution_size_group               small_to_large    class_correct   0.337717          0.246468           0.429585 5.493341e-09 5.954073e-08
  substitution_size_group               small_to_large non_local_recall   0.453377          0.351170           0.553978 1.257488e-10 1.837026e-09
             charge_group           charged_to_neutral    class_correct   0.288156          0.182891           0.391907 3.533179e-06 3.043970e-05
             charge_group           charged_to_neutral non_local_recall   0.469073          0.358893           0.571633 2.840671e-10 3.976939e-09
             charge_group           neutral_to_neutral    class_correct   0.320016          0.230001           0.407070 2.413094e-09 2.895713e-08
             charge_group           neutral_to_neutral non_local_recall   0.522362          0.431261           0.612776 1.247342e-14 3.810064e-13
             charge_group           neutral_to_charged       global_mae  -0.064122         -0.127254          -0.018287 2.339416e-03 1.572087e-02
             charge_group           neutral_to_charged        shell_mae  -0.075194         -0.159669          -0.017826 8.157367e-04 6.229262e-03
             charge_group           neutral_to_charged      mae_shell_0  -0.132988         -0.286422          -0.034156 1.125985e-03 8.407357e-03
             charge_group           neutral_to_charged      mae_shell_2  -0.067181         -0.135407          -0.014047 4.597575e-03 2.970741e-02
             charge_group           neutral_to_charged      mae_shell_4  -0.061516         -0.121870          -0.016412 5.692560e-03 3.477637e-02
             charge_group           neutral_to_charged    class_correct   0.239254          0.081532           0.403466 5.331430e-03 3.317334e-02
             charge_group           neutral_to_charged non_local_recall   0.427486          0.261971           0.588723 1.082870e-04 8.874251e-04
             charge_group charged_to_charged_same_sign    class_correct   0.368812          0.173870           0.570162 3.417969e-03 2.251838e-02
             charge_group charged_to_charged_same_sign non_local_recall   0.433869          0.224460           0.638652 1.708984e-03 1.196289e-02
      charge_binary_group              charge_involved    class_correct   0.307119          0.219209           0.394520 3.969832e-09 4.599530e-08
      charge_binary_group              charge_involved non_local_recall   0.459232          0.368468           0.548053 1.460418e-13 3.287108e-12
      charge_binary_group           no_charge_involved    class_correct   0.320016          0.230001           0.407070 2.413094e-09 2.895713e-08
      charge_binary_group           no_charge_involved non_local_recall   0.522362          0.431261           0.612776 1.247342e-14 3.810064e-13
             glypro_group          gly_or_pro_involved non_local_recall   0.263625          0.064818           0.454273 7.847024e-03 4.545862e-02
             glypro_group                no_gly_or_pro  perturbed_auprc   0.036907          0.017686           0.057602 1.656175e-03 1.183989e-02
             glypro_group                no_gly_or_pro    class_correct   0.334717          0.263620           0.405821 2.993213e-14 8.380995e-13
             glypro_group                no_gly_or_pro non_local_recall   0.516367          0.446302           0.586149 2.866149e-22 4.815130e-20
                gly_group                       no_gly  perturbed_auprc   0.036047          0.016862           0.056359 2.063037e-03 1.414654e-02
                gly_group                       no_gly    class_correct   0.338399          0.269011           0.407072 5.745501e-15 2.757841e-13
                gly_group                       no_gly non_local_recall   0.515511          0.445935           0.583998 1.424032e-22 4.784748e-20
                pro_group                       no_pro  perturbed_auprc   0.033853          0.014441           0.054947 5.074079e-03 3.216774e-02
                pro_group                       no_pro    class_correct   0.316679          0.243666           0.388377 2.453491e-13 4.849252e-12
                pro_group                       no_pro non_local_recall   0.498166          0.429019           0.565476 7.465785e-22 8.361679e-20
secondary_structure_group                        sheet    class_correct   0.335113          0.214957           0.454968 4.253122e-06 3.572622e-05
secondary_structure_group                        sheet non_local_recall   0.509844          0.385608           0.628744 1.188280e-08 1.247694e-07
secondary_structure_group                         loop    class_correct   0.307136          0.225464           0.388495 1.008265e-09 1.302988e-08
secondary_structure_group                         loop non_local_recall   0.467072          0.380200           0.554052 4.060358e-14 1.049446e-12
secondary_structure_group                        helix    class_correct   0.322123          0.210721           0.430534 2.196837e-06 1.942467e-05
secondary_structure_group                        helix non_local_recall   0.513019          0.406594           0.615290 3.321827e-11 5.314923e-10
             length_group                        short    class_correct   0.402843          0.298218           0.508792 1.258616e-08 1.281500e-07
             length_group                        short non_local_recall   0.553586          0.441562           0.661283 3.003185e-10 4.036280e-09
             length_group                         long  perturbed_auprc   0.039501          0.015421           0.066326 6.039527e-03 3.623716e-02
             length_group                         long    class_correct   0.276781          0.186745           0.365415 1.398007e-07 1.342087e-06
             length_group                         long non_local_recall   0.463276          0.371563           0.553739 1.697941e-13 3.565677e-12
         resolution_group               low_resolution    class_correct   0.388979          0.305059           0.471477 5.094582e-13 9.509887e-12
         resolution_group               low_resolution non_local_recall   0.524849          0.441429           0.605740 3.739972e-17 2.513261e-15
         resolution_group              high_resolution  perturbed_auprc   0.045032          0.020003           0.072446 5.338613e-04 4.171568e-03
         resolution_group              high_resolution    class_correct   0.236565          0.155453           0.315291 2.170534e-07 2.025831e-06
         resolution_group              high_resolution non_local_recall   0.431515          0.347858           0.513454 1.467459e-13 3.287108e-12
             domain_group                single_domain       radius_mae  -1.488806         -2.834124          -0.079877 6.332033e-03 3.732567e-02
             domain_group                single_domain    class_correct   0.310766          0.225013           0.396749 4.242841e-09 4.751982e-08
             domain_group                single_domain non_local_recall   0.508615          0.415632           0.596854 1.097741e-14 3.810064e-13
             domain_group                 multi_domain    class_correct   0.363395          0.260718           0.462654 1.472713e-08 1.455386e-07
             domain_group                 multi_domain non_local_recall   0.493898          0.388083           0.595556 7.251569e-11 1.107512e-09
```

```bash
[done] python - <<'PY'
import pandas as pd
p="outputs/call/biological_stratification/v5_seed42_vs_coordinate_residual/pairwise_stratified_diff.csv"
d=pd.read_csv(p)
print(d[d.claimable][["factor","group","metric","mean_diff","bootstrap_ci_low","bootstrap_ci_high","wilcoxon_p","q_bh"]].to_string(index=False))
PY

show outputs when needed
```