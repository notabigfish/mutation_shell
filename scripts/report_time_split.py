from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.train_utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report MuSRNet time-split results")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    load_yaml(args.config)
    audit_dir = Path(args.audit_dir)
    output_dir = Path(args.output_dir)

    summary = load_json(audit_dir / "time_split_summary.json")
    eval_results = {
        split: load_json(output_dir / f"eval_{split}.json")
        for split in ["train", "valid", "test"]
    }

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
        "detection_gap": {
            "test_auprc_minus_valid_auprc": eval_results["test"].get("perturbed_auprc", float("nan")) - eval_results["valid"].get("perturbed_auprc", float("nan"))
        },
        "scientific_question": "Does MuSRNet trained on release_date <= 2022 generalize to 2024-or-later paired WT-mutant structures?",
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
    with (output_dir / "time_split_report.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
