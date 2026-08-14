from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from musrnet.evaluation import compare_cluster_metric, summarize_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict baseline comparison for MuSRNet")
    parser.add_argument("--pred", action="append", required=True, help="model_name=path/to/predictions_test.csv")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    return parser.parse_args()


def parse_pred_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Invalid --pred value: {value}")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = dict(parse_pred_arg(value) for value in args.pred)
    summaries = []
    cluster_frames: list[pd.DataFrame] = []

    for model_name, path in pred_paths.items():
        summary, _, cluster_df = summarize_predictions(pd.read_csv(path))
        summary["model_name"] = model_name
        summaries.append(summary)
        cluster_df.insert(0, "model_name", model_name)
        cluster_frames.append(cluster_df)

    summary_df = pd.DataFrame(summaries)[[
        "model_name",
        "global_mae",
        "shell_mae",
        "cluster_avg_shell_mae",
        "mae_shell_0",
        "mae_shell_1",
        "mae_shell_2",
        "mae_shell_3",
        "mae_shell_4",
        "perturbed_auroc",
        "perturbed_auprc",
        "cluster_avg_auprc",
        "derived_radius_mae",
        "derived_class_macro_f1",
        "cluster_avg_radius_mae",
        "cluster_avg_class_macro_f1",
    ]]
    cluster_metrics_df = pd.concat(cluster_frames, ignore_index=True)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    cluster_metrics_df.to_csv(out_dir / "cluster_metrics.csv", index=False)

    candidate = args.candidate
    baselines = [name for name in pred_paths if name != candidate]
    pairwise_rows = []
    stat_rows = []
    candidate_cluster = cluster_metrics_df[cluster_metrics_df["model_name"] == candidate]
    for baseline_name in baselines:
        baseline_cluster = cluster_metrics_df[cluster_metrics_df["model_name"] == baseline_name]
        merged = candidate_cluster.merge(baseline_cluster, on="cluster_id_30", suffixes=("_candidate", "_baseline"))
        for metric in ["global_mae", "shell_mae", "perturbed_auprc", "derived_radius_mae", "derived_class_macro_f1"]:
            direction = "higher" if metric in {"perturbed_auprc", "derived_class_macro_f1"} else "lower"
            higher_is_better = direction == "higher"
            diff_col = f"{metric}_diff"
            merged_metric = merged[["cluster_id_30", f"{metric}_candidate", f"{metric}_baseline"]].copy()
            merged_metric.insert(0, "candidate", candidate)
            merged_metric.insert(1, "baseline", baseline_name)
            merged_metric.insert(2, "metric", metric)
            merged_metric[diff_col] = merged_metric[f"{metric}_candidate"] - merged_metric[f"{metric}_baseline"]
            pairwise_rows.append(merged_metric)
            stats = compare_cluster_metric(
                candidate_cluster,
                baseline_cluster,
                metric,
                higher_is_better,
                args.seed,
                args.n_bootstrap,
            )
            stats.update({"candidate": candidate, "baseline": baseline_name, "metric": metric, "direction": direction})
            stat_rows.append(stats)

    pairwise_cluster_diffs = pd.concat(pairwise_rows, ignore_index=True) if pairwise_rows else pd.DataFrame()
    pairwise_cluster_diffs.to_csv(out_dir / "pairwise_cluster_diffs.csv", index=False)
    statistical_tests_df = pd.DataFrame(stat_rows)[[
        "candidate",
        "baseline",
        "metric",
        "direction",
        "n_common_clusters",
        "n_improved_clusters",
        "n_worsened_clusters",
        "fraction_improved",
        "mean_diff",
        "median_diff",
        "bootstrap_95ci_low",
        "bootstrap_95ci_high",
        "wilcoxon_pvalue",
    ]]
    statistical_tests_df.to_csv(out_dir / "statistical_tests.csv", index=False)

    summary_lookup = {row["model_name"]: row for row in summaries}
    baseline_summaries = [summary_lookup[name] for name in baselines]
    candidate_summary = summary_lookup[candidate]
    main_claim_passed = (
        candidate_summary["shell_mae"] < min(item["shell_mae"] for item in baseline_summaries)
        and candidate_summary["perturbed_auprc"] > max(item["perturbed_auprc"] for item in baseline_summaries)
    )
    cluster_claim_passed = (
        candidate_summary["cluster_avg_shell_mae"] < min(item["cluster_avg_shell_mae"] for item in baseline_summaries)
        and candidate_summary["cluster_avg_auprc"] > max(item["cluster_avg_auprc"] for item in baseline_summaries)
    )
    best_model_by_metric = {}
    lower_metrics = {"global_mae", "shell_mae", "cluster_avg_shell_mae", "mae_shell_0", "mae_shell_1", "mae_shell_2", "mae_shell_3", "mae_shell_4", "derived_radius_mae", "cluster_avg_radius_mae"}
    for metric in summary_df.columns:
        if metric == "model_name":
            continue
        series = summary_df[["model_name", metric]].dropna()
        if series.empty:
            best_model_by_metric[metric] = None
        elif metric in lower_metrics:
            best_row = series.loc[series[metric].idxmin()]
            best_model_by_metric[metric] = str(best_row["model_name"])
        else:
            best_row = series.loc[series[metric].idxmax()]
            best_model_by_metric[metric] = str(best_row["model_name"])

    strict_summary = {
        "candidate": candidate,
        "baselines": baselines,
        "main_claim_passed": bool(main_claim_passed),
        "cluster_claim_passed": bool(cluster_claim_passed),
        "criterion": {
            "shell_mae": "base_v5 lower than all strict baselines",
            "perturbed_auprc": "base_v5 higher than all strict baselines",
        },
        "best_model_by_metric": best_model_by_metric,
    }
    with (out_dir / "strict_baseline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(strict_summary, handle, indent=2)


if __name__ == "__main__":
    main()
