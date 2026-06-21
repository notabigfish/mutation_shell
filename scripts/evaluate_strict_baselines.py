from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm


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


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def sample_metrics(df: pd.DataFrame) -> dict[str, float]:
    abs_error = (df["pred_displacement"] - df["true_displacement"]).abs()
    metrics = {
        "global_mae": float(abs_error.mean()),
        "perturbed_auroc": safe_auroc(df["true_perturbed"].to_numpy(), df["pred_perturbed_prob"].to_numpy()),
        "perturbed_auprc": safe_auprc(df["true_perturbed"].to_numpy(), df["pred_perturbed_prob"].to_numpy()),
        "derived_radius_mae": float((df["pred_radius"] - df["true_radius"]).abs().mean()),
        "derived_class_macro_f1": float(f1_score(df["true_class"], df["pred_class"], average="macro")),
    }
    shell_values = []
    for shell_idx in range(5):
        shell_df = df[df["shell_id"] == shell_idx]
        value = float((shell_df["pred_displacement"] - shell_df["true_displacement"]).abs().mean()) if not shell_df.empty else float("nan")
        metrics[f"mae_shell_{shell_idx}"] = value
        if not np.isnan(value):
            shell_values.append(value)
    metrics["shell_mae"] = float(np.mean(shell_values)) if shell_values else float("nan")
    return metrics


def summarize_predictions(df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    summary = sample_metrics(df)
    sample_rows = []
    for (cluster_id, sample_id), sample_df in tqdm(df.groupby(["cluster_id_30", "sample_id"], sort=False), total=df.groupby(["cluster_id_30", "sample_id"]).ngroups):
        row = {"cluster_id_30": cluster_id, "sample_id": sample_id}
        row.update(sample_metrics(sample_df))
        sample_rows.append(row)
    sample_metrics_df = pd.DataFrame(sample_rows)
    cluster_metrics_df = sample_metrics_df.groupby("cluster_id_30", dropna=False).mean(numeric_only=True).reset_index()
    summary["cluster_avg_global_mae"] = float(cluster_metrics_df["global_mae"].mean())
    summary["cluster_avg_shell_mae"] = float(cluster_metrics_df["shell_mae"].mean())
    summary["cluster_avg_auprc"] = float(cluster_metrics_df["perturbed_auprc"].mean())
    summary["cluster_avg_radius_mae"] = float(cluster_metrics_df["derived_radius_mae"].mean())
    summary["cluster_avg_class_macro_f1"] = float(cluster_metrics_df["derived_class_macro_f1"].mean())
    return summary, cluster_metrics_df


def compare_metric(candidate: pd.DataFrame, baseline: pd.DataFrame, metric: str, higher_is_better: bool, seed: int, n_bootstrap: int) -> dict[str, Any]:
    merged = candidate[["cluster_id_30", metric]].merge(
        baseline[["cluster_id_30", metric]],
        on="cluster_id_30",
        how="inner",
        suffixes=("_candidate", "_baseline"),
    )
    diff = merged[f"{metric}_candidate"].to_numpy(dtype=float) - merged[f"{metric}_baseline"].to_numpy(dtype=float)
    improved = diff > 0 if higher_is_better else diff < 0
    valid = ~np.isnan(diff)
    diff_valid = diff[valid]
    improved_valid = improved[valid]
    result = {
        "n_common_clusters": int(valid.sum()),
        "n_improved_clusters": int(improved_valid.sum()),
        "n_worsened_clusters": int((~improved_valid).sum()),
        "fraction_improved": float(improved_valid.mean()) if valid.sum() else float("nan"),
        "mean_diff": float(np.nanmean(diff_valid)) if diff_valid.size else float("nan"),
        "median_diff": float(np.nanmedian(diff_valid)) if diff_valid.size else float("nan"),
        "bootstrap_95ci_low": float("nan"),
        "bootstrap_95ci_high": float("nan"),
        "wilcoxon_pvalue": float("nan"),
    }
    if diff_valid.size >= 1:
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, diff_valid.size, size=diff_valid.size)
            boot.append(float(np.nanmean(diff_valid[idx])))
        ci = np.nanpercentile(np.asarray(boot, dtype=float), [2.5, 97.5])
        result["bootstrap_95ci_low"] = float(ci[0])
        result["bootstrap_95ci_high"] = float(ci[1])
    if diff_valid.size >= 2 and not np.allclose(diff_valid, 0.0):
        try:
            result["wilcoxon_pvalue"] = float(wilcoxon(diff_valid).pvalue)
        except Exception:
            result["wilcoxon_pvalue"] = float("nan")
    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = dict(parse_pred_arg(value) for value in args.pred)
    summaries = []
    cluster_frames: list[pd.DataFrame] = []

    for model_name, path in pred_paths.items():
        df = pd.read_csv(path)
        summary, cluster_df = summarize_predictions(df)
        summary["model_name"] = model_name
        summaries.append(summary)
        cluster_df = cluster_df.copy()
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
            stats = compare_metric(candidate_cluster, baseline_cluster, metric, higher_is_better, args.seed, args.n_bootstrap)
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
