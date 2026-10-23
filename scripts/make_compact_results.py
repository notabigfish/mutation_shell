from pathlib import Path
import shutil

SRC = Path("outputs/call")
DST = Path("compact_results")

if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)

def cp(path):
    src = SRC / path
    if not src.exists():
        print("MISSING:", src)
        return
    dst = DST / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def cp_dir_files(dirname, names):
    for name in names:
        cp(Path(dirname) / name)

# 1. Strict baseline comparisons: these already contain manuscript-level results
for seed in [42, 101, 668]:
    d = f"strict_baseline_comparison_base_v5_seed{seed}"
    cp_dir_files(d, [
        "summary_metrics.csv",
        "cluster_metrics.csv",
        "pairwise_cluster_diffs.csv",
        "statistical_tests.csv",
        "strict_baseline_summary.json",
    ])

# Main run compact outputs
for seed in [42, 101, 668]:
    d = f"base_v5_seed{seed}"
    cp_dir_files(d, [
        "eval_test.json",
        "eval_valid.json",
        "cluster_metrics_test.csv",
        "cluster_metrics_valid.csv",
    ])

# Baseline eval summaries
for model in [
    "zero_response",
    "global_mean",
    "shell_mean",
    "mutation_type_shell_mean",
    "esm_mlp",
    "geometry_gnn",
    "coordinate_residual",
]:
    cp_dir_files(model, [
        "eval_test.json",
        "baseline_stats.json",
    ])

# 2. Mechanism closure
for d in [
    "mechanism_closure_seed42",
    "context_ablation_valid/no_context_419952",
]:
    cp_dir_files(d, [
        "summary_metrics.csv",
        "cluster_metrics.csv",
        "pairwise_cluster_diffs.csv",
        "statistical_tests.csv",
        "strict_baseline_summary.json",
    ])

for d in [
    "base_v5_direct_seed42",
    "base_v5_pure_hurdle_seed42",
    "base_v5_global_loss_seed42",
]:
    cp_dir_files(d, [
        "eval_test.json",
        "cluster_metrics_test.csv",
    ])

# 3. Corrected counterfactuals
for seed in range(10):
    d = f"base_v5_seed42/counterfactual_corrected_seed{seed}"
    cp_dir_files(d, [
        "counterfactual_summary.csv",
        "counterfactual_summary.json",

        "metrics_original.json",
        "metrics_remove_mutation_context.json",
        "metrics_reverse_mutation.json",
        "metrics_shuffle_mutant_aa.json",
        "metrics_shuffle_site.json",
        "metrics_wt_wt_negative.json",

        "cluster_metrics_original.csv",
        "cluster_metrics_remove_mutation_context.csv",
        "cluster_metrics_reverse_mutation.csv",
        "cluster_metrics_shuffle_mutant_aa.csv",
        "cluster_metrics_shuffle_site.csv",
        "cluster_metrics_wt_wt_negative.csv",
    ])

# 4. Threshold sensitivity
cp("threshold_sensitivity.csv")
cp("threshold_sensitivity_valid.csv")

# 5. Alignment sensitivity
cp_dir_files("alignment_sensitivity/final", [
    "per_variant_metrics.csv",
    "summary.csv",
    "summary.json",
])

for ref in [
    "base_v5_seed42",
    "base_v5_seed101",
    "base_v5_seed668",
    "kabsch_all",
    "kabsch_exclude_4A",
    "kabsch_exclude_8A",
]:
    d = f"alignment_sensitivity/cluster_compare_ref_{ref}"
    cp_dir_files(d, [
        "cluster_metrics_wide.csv",
        "pairwise_cluster_diff.csv",
        "summary.json",
    ])

for d in [
    "alignment_sensitivity/label_compare_kabsch_exclude_4A_vs_all",
    "alignment_sensitivity/label_compare_kabsch_exclude_8A_vs_all",
]:
    cp_dir_files(d, ["summary.json"])

for variant in [
    "kabsch_all",
    "kabsch_exclude_4A",
    "kabsch_exclude_8A",
]:
    cp_dir_files(
        f"alignment_sensitivity/{variant}",
        ["label_stats.json", "failed_samples.csv"]
    )

# 6. Biological stratification
for seed in [42, 101, 668]:
    for ref in ["geometry_gnn", "coordinate_residual"]:
        d = f"biological_stratification/v5_seed{seed}_vs_{ref}"
        cp_dir_files(d, [
            "stratified_metrics_long.csv",
            "stratified_metrics_wide.csv",
            "pairwise_stratified_diff.csv",
            "summary.json",
            "warnings.txt",
        ])

# 7. Time split
cp_dir_files("base_v5_time", [
    "time_split_report.json",
    "time_split_report.md",
    "eval_train.json",
    "eval_valid.json",
    "eval_test.json",
    "cluster_metrics_train.csv",
    "cluster_metrics_valid.csv",
    "cluster_metrics_test.csv",
])

cp_dir_files("geometry_gnn_time", [
    "eval_valid.json",
    "eval_test.json",
    "cluster_metrics_valid.csv",
    "cluster_metrics_test.csv",
])

for d in ["shell_mean_time", "mutation_type_shell_mean_time"]:
    cp_dir_files(d, [
        "baseline_stats.json",
        "eval_valid.json",
        "eval_test.json",
    ])

cp_dir_files("time_split_comparison", [
    "summary_metrics.csv",
    "cluster_metrics.csv",
    "pairwise_cluster_diffs.csv",
    "statistical_tests.csv",
    "strict_baseline_summary.json",
])

print("Compact results written to:", DST)
