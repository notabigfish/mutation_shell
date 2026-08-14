from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.train_utils import load_yaml
from musrnet.evaluation import summarize_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report MuSRNet time-split results")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def label_shift(path: Path) -> dict[str, float]:
    n=0; pos=0.0; disp=0.0
    for df in pd.read_csv(path,usecols=["true_perturbed","true_displacement"],chunksize=1000000):
        n+=len(df); pos+=float(df["true_perturbed"].sum()); disp+=float(df["true_displacement"].sum())
    return {"perturbed_fraction":pos/n,"mean_true_displacement":disp/n}

def main() -> None:
    args = parse_args()
    load_yaml(args.config)
    audit_dir = Path(args.audit_dir)
    output_dir = Path(args.output_dir)

    summary = load_json(audit_dir / "time_split_summary.json")
    eval_results = {}
    for split in ["train", "valid", "test"]:
        metrics, sample_metrics, cluster_metrics = summarize_predictions(pd.read_csv(output_dir / f"predictions_{split}.csv"))
        eval_results[split] = metrics
        with (output_dir / f"eval_{split}.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        sample_metrics.to_csv(output_dir / f"sample_metrics_{split}.csv", index=False)
        cluster_metrics.to_csv(output_dir / f"cluster_metrics_{split}.csv", index=False)
    shift={s:label_shift(output_dir/f"predictions_{s}.csv") for s in ["valid","test"]}
    for s in shift:
        p=shift[s]["perturbed_fraction"]
        shift[s]["auprc_lift_vs_prevalence"]=eval_results[s]["perturbed_auprc"]/p if p else float("nan")
    report = {
        "config": str(args.config),
        "split_rule": summary["split_rule"],
        "n_samples": summary["n_samples"],
        "n_clusters": summary["n_clusters"],
        "sample_year_range": summary["sample_year_range"],
        "cluster_year_range": summary["cluster_year_range"],
        "metrics": eval_results,
        "generalization_gap": {
            "valid_shell_mae_minus_train_shell_mae": eval_results["valid"].get("shell_mae", float("nan")) - eval_results["train"].get("shell_mae", float("nan")),
            "test_shell_mae_minus_train_shell_mae": eval_results["test"].get("shell_mae", float("nan")) - eval_results["train"].get("shell_mae", float("nan")),
            "test_shell_mae_minus_valid_shell_mae": eval_results["test"].get("shell_mae", float("nan")) - eval_results["valid"].get("shell_mae", float("nan")),
        },
        "detection_gap":{"test_auprc_minus_valid_auprc":eval_results["test"]["perturbed_auprc"]-eval_results["valid"]["perturbed_auprc"]},
        "label_shift":shift,
        "scientific_question":f"Does MuSRNet generalize from {summary['split_rule']['train']} to {summary['split_rule']['test']} under cluster-safe temporal splitting?",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "time_split_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    lines = [
        "# Time-Split Report",
        "",
        f"Config: `{args.config}`",
        "",
        "| split | year rule | samples | clusters | global_mae | shell_mae | cluster_avg_shell_mae | AUROC | AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    year_rule_map = {
        "train": summary["split_rule"]["train"],
        "valid": summary["split_rule"]["valid"],
        "test": summary["split_rule"]["test"],
    }
    for split in ["train", "valid", "test"]:
        metrics = eval_results[split]
        lines.append(
            f"| {split} | {year_rule_map[split]} | {summary['n_samples'][split]} | {summary['n_clusters'][split]} | "
            f"{metrics.get('global_mae', float('nan')):.4f} | {metrics.get('shell_mae', float('nan')):.4f} | "
            f"{metrics.get('cluster_avg_shell_mae', float('nan')):.4f} | {metrics.get('perturbed_auroc', float('nan')):.4f} | "
            f"{metrics.get('perturbed_auprc', float('nan')):.4f} |"
        )
    lines.extend(
        [
            "",
            "Generalization gap:",
            f"- valid_shell_mae - train_shell_mae = {report['generalization_gap']['valid_shell_mae_minus_train_shell_mae']:.4f}",
            f"- test_shell_mae - train_shell_mae = {report['generalization_gap']['test_shell_mae_minus_train_shell_mae']:.4f}",
            f"- test_shell_mae - valid_shell_mae = {report['generalization_gap']['test_shell_mae_minus_valid_shell_mae']:.4f}",
            "",
            "Detection gap:",
            f"- test_auprc - valid_auprc = {report['detection_gap']['test_auprc_minus_valid_auprc']:.4f}",
        ]
    )
    lines.extend([
        "",
        "Label shift:",
        f"- valid perturbed_fraction = {shift['valid']['perturbed_fraction']:.4f}",
        f"- test perturbed_fraction = {shift['test']['perturbed_fraction']:.4f}",
        f"- valid AUPRC/prevalence = {shift['valid']['auprc_lift_vs_prevalence']:.3f}",
        f"- test AUPRC/prevalence = {shift['test']['auprc_lift_vs_prevalence']:.3f}",
    ])
    with (output_dir / "time_split_report.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
