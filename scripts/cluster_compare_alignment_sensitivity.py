from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.statistics import paired_bootstrap_ci, wilcoxon_signed_rank_pvalue
from musrnet.train_utils import save_json

COLUMN_ALIASES = {
    "sample_id": ["sample_id"],
    "cluster_id_30": ["cluster_id_30", "cluster_id", "cluster"],
    "shell_id": ["shell_id", "shell"],
    "true_disp": ["true_disp", "true_displacement"],
    "pred_disp": ["pred_disp", "pred_displacement"],
    "true_perturbed": ["true_perturbed"],
    "pred_perturbed_prob": ["pred_perturbed_prob"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster-level paired comparison across alignment variants")
    parser.add_argument("--pred", action="append", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def parse_pred_arg(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise ValueError(f"Invalid --pred value: {arg}")
    name, path_str = arg.split("=", 1)
    return name.strip(), PROJECT_ROOT / path_str.strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = target
                break
    df = df.rename(columns=rename_map).copy()
    required = ["sample_id", "cluster_id_30", "shell_id", "true_disp", "pred_disp", "true_perturbed", "pred_perturbed_prob"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def safe_au_metric(fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return float("nan")


def compute_cluster_metrics(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for cluster_id, cluster_df in df.groupby("cluster_id_30", sort=False):
        errors = np.abs(cluster_df["pred_disp"].to_numpy(dtype=np.float64) - cluster_df["true_disp"].to_numpy(dtype=np.float64))
        row = {
            "cluster_id_30": str(cluster_id),
            "n_samples": int(cluster_df["sample_id"].nunique()),
            "n_residues": int(len(cluster_df)),
            "global_mae_cluster_mean": float(errors.mean()),
            "perturbed_auroc_cluster_mean": safe_au_metric(
                roc_auc_score,
                cluster_df["true_perturbed"].to_numpy(dtype=np.int64),
                cluster_df["pred_perturbed_prob"].to_numpy(dtype=np.float64),
            ),
            "perturbed_auprc_cluster_mean": safe_au_metric(
                average_precision_score,
                cluster_df["true_perturbed"].to_numpy(dtype=np.int64),
                cluster_df["pred_perturbed_prob"].to_numpy(dtype=np.float64),
            ),
        }
        shell_values = []
        for shell_idx in range(5):
            mask = cluster_df["shell_id"].to_numpy() == shell_idx
            key = f"mae_shell_{shell_idx}_cluster_mean"
            row[key] = float(errors[mask].mean()) if mask.any() else float("nan")
            if mask.any():
                shell_values.append(row[key])
        row["shell_mae_cluster_mean"] = float(np.mean(shell_values)) if shell_values else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_frames = []
    by_name = {}
    for arg in args.pred:
        name, path = parse_pred_arg(arg)
        df = normalize_columns(pd.read_csv(path))
        cluster_df = compute_cluster_metrics(df)
        cluster_df.insert(0, "alignment_variant", name)
        cluster_frames.append(cluster_df)
        by_name[name] = cluster_df

    cluster_metrics_wide = pd.concat(cluster_frames, ignore_index=True)
    cluster_metrics_wide.to_csv(out_dir / "cluster_metrics_wide.csv", index=False)

    reference_df = by_name[args.reference]
    pairwise_rows = []
    summary_rows = []
    for name, candidate_df in by_name.items():
        if name == args.reference:
            continue
        merged = reference_df.merge(candidate_df, on="cluster_id_30", suffixes=("_reference", "_candidate"))
        diff_shell = merged["shell_mae_cluster_mean_candidate"].to_numpy(dtype=np.float64) - merged["shell_mae_cluster_mean_reference"].to_numpy(dtype=np.float64)
        diff_auprc = merged["perturbed_auprc_cluster_mean_candidate"].to_numpy(dtype=np.float64) - merged["perturbed_auprc_cluster_mean_reference"].to_numpy(dtype=np.float64)
        merged["diff_shell_mae"] = diff_shell
        merged["diff_perturbed_auprc"] = diff_auprc
        merged["improved_shell_mae"] = diff_shell < 0
        pairwise_rows.append(merged.assign(reference=args.reference, candidate=name))

        weights_samples = merged["n_samples_candidate"].to_numpy(dtype=np.float64)
        weights_res = merged["n_residues_candidate"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "reference": args.reference,
                "candidate": name,
                "n_common_clusters": int(len(merged)),
                "n_clusters_shell_mae_improved": int(np.sum(diff_shell < 0)),
                "n_clusters_shell_mae_worsened": int(np.sum(diff_shell > 0)),
                "fraction_clusters_shell_mae_improved": float(np.mean(diff_shell < 0)) if len(diff_shell) else float("nan"),
                "mean_diff_shell_mae": safe_mean(diff_shell),
                "median_diff_shell_mae": float(np.nanmedian(diff_shell)) if len(diff_shell) else float("nan"),
                "mean_diff_perturbed_auprc": safe_mean(diff_auprc),
                "median_diff_perturbed_auprc": float(np.nanmedian(diff_auprc)) if len(diff_auprc) else float("nan"),
                "weighted_mean_diff_shell_mae_by_n_samples": float(np.average(diff_shell, weights=weights_samples)) if weights_samples.sum() else float("nan"),
                "weighted_mean_diff_shell_mae_by_n_residues": float(np.average(diff_shell, weights=weights_res)) if weights_res.sum() else float("nan"),
                "shell_mae_bootstrap_ci_low": paired_bootstrap_ci(diff_shell)[0],
                "shell_mae_bootstrap_ci_high": paired_bootstrap_ci(diff_shell)[1],
                "shell_mae_wilcoxon_pvalue": wilcoxon_signed_rank_pvalue(diff_shell),
                "auprc_bootstrap_ci_low": paired_bootstrap_ci(diff_auprc)[0],
                "auprc_bootstrap_ci_high": paired_bootstrap_ci(diff_auprc)[1],
                "auprc_wilcoxon_pvalue": wilcoxon_signed_rank_pvalue(diff_auprc),
            }
        )

    pairwise_df = pd.concat(pairwise_rows, ignore_index=True) if pairwise_rows else pd.DataFrame()
    pairwise_df.to_csv(out_dir / "pairwise_cluster_diff.csv", index=False)
    save_json(out_dir / "summary.json", {"rows": summary_rows})


if __name__ == "__main__":
    main()
