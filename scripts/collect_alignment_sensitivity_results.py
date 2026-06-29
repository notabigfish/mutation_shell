from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.train_utils import load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect alignment-sensitivity training and evaluation results")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def parse_run_arg(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise ValueError(f"Invalid --run value: {arg}")
    name, path_str = arg.split("=", 1)
    return name.strip(), PROJECT_ROOT / path_str.strip()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def best_eval_shell_mae_from_trainer_state(path: Path) -> float:
    state = read_json(path)
    values = [row.get("eval_shell_mae") for row in state.get("log_history", []) if isinstance(row, dict) and "eval_shell_mae" in row]
    values = [float(v) for v in values if v is not None]
    return float(min(values)) if values else float("nan")


def extract_label_stats(variant: str) -> dict:
    path = PROJECT_ROOT / "results" / "alignment_sensitivity" / variant / "label_stats.json"
    if not path.exists():
        return {}
    return read_json(path)


def collect_one(variant: str, run_dir: Path) -> dict:
    config_path = run_dir / "config_used.yaml"
    trainer_state_path = run_dir / "trainer_state.json"
    eval_test_path = run_dir / "eval_test.json"
    predictions_path = run_dir / "predictions_test.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing {predictions_path}. Run the existing evaluation script first, for example: "
            f"python scripts/evaluate.py --config {config_path} --checkpoint <checkpoint>"
        )

    config = load_yaml(config_path) if config_path.exists() else {}
    eval_test = read_json(eval_test_path) if eval_test_path.exists() else {}
    label_stats = extract_label_stats(variant)
    row = {
        "alignment_variant": variant,
        "best_eval_shell_mae": best_eval_shell_mae_from_trainer_state(trainer_state_path) if trainer_state_path.exists() else float("nan"),
        "test_global_mae": eval_test.get("global_mae", float("nan")),
        "test_shell_mae": eval_test.get("shell_mae", float("nan")),
        "test_cluster_avg_shell_mae": eval_test.get("cluster_avg_shell_mae", float("nan")),
        "test_mae_shell_0": eval_test.get("mae_shell_0", float("nan")),
        "test_mae_shell_1": eval_test.get("mae_shell_1", float("nan")),
        "test_mae_shell_2": eval_test.get("mae_shell_2", float("nan")),
        "test_mae_shell_3": eval_test.get("mae_shell_3", float("nan")),
        "test_mae_shell_4": eval_test.get("mae_shell_4", float("nan")),
        "test_perturbed_auroc": eval_test.get("perturbed_auroc", float("nan")),
        "test_perturbed_auprc": eval_test.get("perturbed_auprc", float("nan")),
        "test_radius_mae": eval_test.get("radius_mae", float("nan")),
        "test_class_macro_f1": eval_test.get("class_macro_f1", float("nan")),
        "mean_alignment_rmsd": label_stats.get("mean_alignment_rmsd", float("nan")),
        "mean_shell0_label_disp": label_stats.get("mean_shell_displacement", {}).get("0", float("nan")),
        "mean_shell1_label_disp": label_stats.get("mean_shell_displacement", {}).get("1", float("nan")),
        "mean_shell4_label_disp": label_stats.get("mean_shell_displacement", {}).get("4", float("nan")),
        "perturbed_fraction_shell0": label_stats.get("perturbed_fraction_by_shell", {}).get("0", float("nan")),
        "perturbed_fraction_shell1": label_stats.get("perturbed_fraction_by_shell", {}).get("1", float("nan")),
        "perturbed_fraction_shell4": label_stats.get("perturbed_fraction_by_shell", {}).get("4", float("nan")),
        "output_dir": str(run_dir),
        "prediction_csv": str(predictions_path),
        "config_used": str(config_path) if config_path.exists() else None,
    }
    if config:
        row["alignment_variant_config"] = config.get("data", {}).get("alignment_variant")
    return row


def build_interpretation(rows: list[dict]) -> dict:
    required = ["test_shell_mae", "test_mae_shell_0", "test_mae_shell_1", "test_perturbed_auprc", "test_mae_shell_4", "test_cluster_avg_shell_mae"]
    if any(any(np.isnan(float(row.get(metric, np.nan))) for metric in required) for row in rows):
        preferred = None
    else:
        scored = sorted(rows, key=lambda row: (row["test_shell_mae"], row["test_mae_shell_0"], row["test_mae_shell_1"], -row["test_perturbed_auprc"]))
        preferred = scored[0]["alignment_variant"] if scored else None
    statements: list[str] = []
    by_name = {row["alignment_variant"]: row for row in rows}
    if preferred == "kabsch_exclude_4A":
        row_4 = by_name["kabsch_exclude_4A"]
        row_all = by_name.get("kabsch_all")
        if row_all and row_4["test_mae_shell_0"] <= row_all["test_mae_shell_0"] and row_4["test_perturbed_auprc"] >= row_all["test_perturbed_auprc"]:
            statements.append("Excluding the immediate mutation neighborhood gives cleaner local-response labels.")
    if preferred == "kabsch_all":
        statements.append("Local-neighborhood exclusion is not supported by the current data.")
    row_8 = by_name.get("kabsch_exclude_8A")
    row_4 = by_name.get("kabsch_exclude_4A")
    if row_8 and row_4:
        if row_8["mean_shell1_label_disp"] > row_4["mean_shell1_label_disp"] and row_8["test_cluster_avg_shell_mae"] > row_4["test_cluster_avg_shell_mae"]:
            statements.append("Excluding too broad a neighborhood may inject global alignment noise.")
    row_tm = by_name.get("tmalign")
    if row_tm and row_4:
        if row_tm["test_mae_shell_0"] >= row_4["test_mae_shell_0"] and row_tm["test_perturbed_auprc"] <= row_4["test_perturbed_auprc"]:
            statements.append("TM-align is useful as a sanity check but not the primary label-alignment rule for matched single-mutant pairs.")
    return {
        "primary_question": "whether mutation-neighborhood exclusion gives cleaner local-response labels",
        "decision_rule": {
            "preferred_alignment": preferred,
            "criteria": [
                "lowest test shell_mae",
                "lowest test mae_shell_0 and mae_shell_1",
                "competitive or improved test perturbed_auprc",
                "no large degradation in mae_shell_4",
                "better or comparable cluster_avg_shell_mae",
            ],
        },
        "interpretation": statements,
    }


def main() -> None:
    args = parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [collect_one(*parse_run_arg(arg)) for arg in args.run]
    df = pd.DataFrame(rows)
    summary_columns = [
        "alignment_variant",
        "best_eval_shell_mae",
        "test_global_mae",
        "test_shell_mae",
        "test_cluster_avg_shell_mae",
        "test_mae_shell_0",
        "test_mae_shell_1",
        "test_mae_shell_2",
        "test_mae_shell_3",
        "test_mae_shell_4",
        "test_perturbed_auroc",
        "test_perturbed_auprc",
        "test_radius_mae",
        "test_class_macro_f1",
        "mean_alignment_rmsd",
        "mean_shell0_label_disp",
        "mean_shell1_label_disp",
        "mean_shell4_label_disp",
        "perturbed_fraction_shell0",
        "perturbed_fraction_shell1",
        "perturbed_fraction_shell4",
    ]
    df[summary_columns].to_csv(out_dir / "summary.csv", index=False)
    df.to_csv(out_dir / "per_variant_metrics.csv", index=False)
    payload = {
        "summary_rows": rows,
        **build_interpretation(rows),
    }
    save_json(out_dir / "summary.json", payload)


if __name__ == "__main__":
    main()
