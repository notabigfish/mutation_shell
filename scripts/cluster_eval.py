from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm


COLUMN_ALIASES = {
    "true_displacement": ["true_displacement", "true_disp", "y_disp"],
    "pred_displacement": ["pred_displacement", "pred_disp", "disp"],
    "cluster_id_30": ["cluster_id_30", "cluster_id", "cluster"],
    "shell_id": ["shell_id", "shell"],
    "sample_id": ["sample_id", "protein_id"],
    "true_perturbed": ["true_perturbed", "y_perturbed"],
    "pred_perturbed_prob": ["pred_perturbed_prob", "perturbed_prob", "pred_prob"],
}
REQUIRED_COLUMNS = [
    "sample_id",
    "cluster_id_30",
    "shell_id",
    "true_displacement",
    "pred_displacement",
    "true_perturbed",
    "pred_perturbed_prob",
]
METRIC_COLUMNS = [
    "global_mae",
    "shell_mae",
    "mae_shell_0",
    "mae_shell_1",
    "mae_shell_2",
    "mae_shell_3",
    "mae_shell_4",
    "perturbed_auroc",
    "perturbed_auprc",
]


def parse_pred_arg(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise ValueError(f"Invalid --pred value '{arg}'. Expected model_name=path/to/predictions.csv")
    model_name, path_str = arg.split("=", 1)
    model_name = model_name.strip()
    path = Path(path_str.strip())
    if not model_name:
        raise ValueError(f"Invalid --pred value '{arg}'. Model name is empty")
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    return model_name, path


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for normalized, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = normalized
                break
    df = df.rename(columns=rename_map).copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["sample_id"] = df["sample_id"].astype(str)
    df["cluster_id_30"] = df["cluster_id_30"].astype(str)
    df["shell_id"] = pd.to_numeric(df["shell_id"], errors="coerce")
    df["true_displacement"] = pd.to_numeric(df["true_displacement"], errors="coerce")
    df["pred_displacement"] = pd.to_numeric(df["pred_displacement"], errors="coerce")
    df["true_perturbed"] = pd.to_numeric(df["true_perturbed"], errors="coerce")
    df["pred_perturbed_prob"] = pd.to_numeric(df["pred_perturbed_prob"], errors="coerce")

    numeric_required = [
        "shell_id",
        "true_displacement",
        "pred_displacement",
        "true_perturbed",
        "pred_perturbed_prob",
    ]
    df = df.dropna(subset=numeric_required).copy()
    df["shell_id"] = df["shell_id"].astype(int)
    df["true_perturbed"] = df["true_perturbed"].astype(int)
    return df


def safe_mean(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if roc_auc_score is None:
        return float("nan")
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if average_precision_score is None:
        return float("nan")
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def compute_sample_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for sample_id, sample_df in tqdm(df.groupby("sample_id", sort=False), total=df["sample_id"].nunique()):
        errors = np.abs(
            sample_df["pred_displacement"].to_numpy(dtype=np.float64)
            - sample_df["true_displacement"].to_numpy(dtype=np.float64)
        )
        cluster_id = str(sample_df["cluster_id_30"].iloc[0])
        row = {
            "sample_id": str(sample_id),
            "cluster_id_30": cluster_id,
            "global_mae": float(errors.mean()) if errors.size else float("nan"),
            "perturbed_auroc": safe_auroc(
                sample_df["true_perturbed"].to_numpy(dtype=np.int64),
                sample_df["pred_perturbed_prob"].to_numpy(dtype=np.float64),
            ),
            "perturbed_auprc": safe_auprc(
                sample_df["true_perturbed"].to_numpy(dtype=np.int64),
                sample_df["pred_perturbed_prob"].to_numpy(dtype=np.float64),
            ),
            "n_residues": int(len(sample_df)),
        }
        shell_values = []
        for shell_idx in range(5):
            shell_mask = sample_df["shell_id"].to_numpy() == shell_idx
            metric_name = f"mae_shell_{shell_idx}"
            if shell_mask.any():
                row[metric_name] = float(errors[shell_mask].mean())
                shell_values.append(row[metric_name])
            else:
                row[metric_name] = float("nan")
        row["shell_mae"] = safe_mean(shell_values)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_cluster_metrics(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for cluster_id, cluster_df in tqdm(sample_metrics.groupby("cluster_id_30", sort=False), total=sample_metrics["cluster_id_30"].nunique()):
        row = {
            "cluster_id_30": str(cluster_id),
            "n_samples": int(cluster_df["sample_id"].nunique()),
            "n_residues": int(cluster_df["n_residues"].sum()),
        }
        for metric in METRIC_COLUMNS:
            values = cluster_df[metric].to_numpy(dtype=np.float64)
            valid_mask = ~np.isnan(values)
            row[metric] = safe_mean(values)
            row[f"n_valid_samples_for_{metric}"] = int(valid_mask.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def make_long_table(cluster_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for record in cluster_metrics.to_dict(orient="records"):
        for metric in METRIC_COLUMNS:
            rows.append(
                {
                    "model_name": record["model_name"],
                    "cluster_id_30": record["cluster_id_30"],
                    "metric": metric,
                    "value": record[metric],
                    "n_samples": record["n_samples"],
                    "n_residues": record["n_residues"],
                }
            )
    return pd.DataFrame(rows)


def make_pairwise_table(cluster_metrics: pd.DataFrame, reference: str, candidate: str) -> pd.DataFrame:
    ref = cluster_metrics[cluster_metrics["model_name"] == reference].copy()
    cand = cluster_metrics[cluster_metrics["model_name"] == candidate].copy()
    merged = ref.merge(
        cand,
        on="cluster_id_30",
        how="inner",
        suffixes=("_reference", "_candidate"),
    )
    if merged.empty:
        return merged

    merged["diff_global_mae"] = merged["global_mae_candidate"] - merged["global_mae_reference"]
    merged["diff_shell_mae"] = merged["shell_mae_candidate"] - merged["shell_mae_reference"]
    merged["diff_perturbed_auprc"] = merged["perturbed_auprc_candidate"] - merged["perturbed_auprc_reference"]
    merged["diff_perturbed_auroc"] = merged["perturbed_auroc_candidate"] - merged["perturbed_auroc_reference"]
    merged["improved_global_mae"] = merged["diff_global_mae"] < 0
    merged["improved_shell_mae"] = merged["diff_shell_mae"] < 0
    merged["improved_perturbed_auprc"] = merged["diff_perturbed_auprc"] > 0

    pairwise = merged.rename(
        columns={
            "global_mae_reference": "reference_global_mae",
            "global_mae_candidate": "candidate_global_mae",
            "shell_mae_reference": "reference_shell_mae",
            "shell_mae_candidate": "candidate_shell_mae",
            "perturbed_auprc_reference": "reference_perturbed_auprc",
            "perturbed_auprc_candidate": "candidate_perturbed_auprc",
            "perturbed_auroc_reference": "reference_perturbed_auroc",
            "perturbed_auroc_candidate": "candidate_perturbed_auroc",
        }
    )
    pairwise = pairwise[
        [
            "cluster_id_30",
            "n_samples_reference",
            "n_samples_candidate",
            "n_residues_reference",
            "n_residues_candidate",
            "reference_global_mae",
            "candidate_global_mae",
            "diff_global_mae",
            "reference_shell_mae",
            "candidate_shell_mae",
            "diff_shell_mae",
            "reference_perturbed_auprc",
            "candidate_perturbed_auprc",
            "diff_perturbed_auprc",
            "reference_perturbed_auroc",
            "candidate_perturbed_auroc",
            "diff_perturbed_auroc",
            "improved_shell_mae",
            "improved_global_mae",
            "improved_perturbed_auprc",
        ]
    ]
    return pairwise.sort_values("diff_shell_mae", ascending=True).reset_index(drop=True)


def build_summary(
    cluster_metrics: pd.DataFrame,
    pairwise: pd.DataFrame | None,
    reference: str | None,
    candidate: str | None,
) -> dict:
    summary = {
        "models": sorted(cluster_metrics["model_name"].unique().tolist()),
        "reference": reference,
        "candidate": candidate,
        "model_summary": {},
    }
    for model_name, model_df in cluster_metrics.groupby("model_name", sort=False):
        summary["model_summary"][model_name] = {
            "global_mae_cluster_mean": safe_mean(model_df["global_mae"]),
            "shell_mae_cluster_mean": safe_mean(model_df["shell_mae"]),
            "mae_shell_0_cluster_mean": safe_mean(model_df["mae_shell_0"]),
            "mae_shell_1_cluster_mean": safe_mean(model_df["mae_shell_1"]),
            "mae_shell_2_cluster_mean": safe_mean(model_df["mae_shell_2"]),
            "mae_shell_3_cluster_mean": safe_mean(model_df["mae_shell_3"]),
            "mae_shell_4_cluster_mean": safe_mean(model_df["mae_shell_4"]),
            "perturbed_auroc_cluster_mean": safe_mean(model_df["perturbed_auroc"]),
            "perturbed_auprc_cluster_mean": safe_mean(model_df["perturbed_auprc"]),
            "n_clusters": int(len(model_df)),
            "n_samples": int(model_df["n_samples"].sum()),
            "n_residues": int(model_df["n_residues"].sum()),
        }

    if pairwise is not None and not pairwise.empty:
        diff_shell = pairwise["diff_shell_mae"].to_numpy(dtype=np.float64)
        diff_global = pairwise["diff_global_mae"].to_numpy(dtype=np.float64)
        diff_auprc = pairwise["diff_perturbed_auprc"].to_numpy(dtype=np.float64)
        diff_auroc = pairwise["diff_perturbed_auroc"].to_numpy(dtype=np.float64)
        improved = pairwise["improved_shell_mae"].fillna(False)
        worsened = pairwise["diff_shell_mae"] > 0
        tied = ~(improved | worsened)

        sample_weights = pairwise["n_samples_reference"].to_numpy(dtype=np.float64)
        residue_weights = pairwise["n_residues_reference"].to_numpy(dtype=np.float64)
        best_cluster = pairwise.iloc[pairwise["diff_shell_mae"].idxmin()]
        worst_cluster = pairwise.iloc[pairwise["diff_shell_mae"].idxmax()]
        summary["pairwise_summary"] = {
            "n_common_clusters": int(len(pairwise)),
            "n_clusters_shell_mae_improved": int(improved.sum()),
            "n_clusters_shell_mae_worsened": int(worsened.sum()),
            "n_clusters_shell_mae_tied": int(tied.sum()),
            "fraction_clusters_shell_mae_improved": float(improved.mean()) if len(pairwise) else float("nan"),
            "mean_diff_shell_mae": safe_mean(diff_shell),
            "median_diff_shell_mae": float(np.nanmedian(diff_shell)) if len(diff_shell) else float("nan"),
            "mean_diff_global_mae": safe_mean(diff_global),
            "median_diff_global_mae": float(np.nanmedian(diff_global)) if len(diff_global) else float("nan"),
            "mean_diff_perturbed_auprc": safe_mean(diff_auprc),
            "median_diff_perturbed_auprc": float(np.nanmedian(diff_auprc)) if len(diff_auprc) else float("nan"),
            "mean_diff_perturbed_auroc": safe_mean(diff_auroc),
            "median_diff_perturbed_auroc": float(np.nanmedian(diff_auroc)) if len(diff_auroc) else float("nan"),
            "weighted_mean_diff_shell_mae_by_n_samples": float(np.average(diff_shell, weights=sample_weights)),
            "weighted_mean_diff_shell_mae_by_n_residues": float(np.average(diff_shell, weights=residue_weights)),
            "best_improved_cluster_by_shell_mae": str(best_cluster["cluster_id_30"]),
            "worst_worsened_cluster_by_shell_mae": str(worst_cluster["cluster_id_30"]),
        }
    return summary


def maybe_make_plots(pairwise: pd.DataFrame, out_dir: Path) -> None:
    if plt is None or pairwise.empty:
        return
    fig = plt.figure()
    plt.hist(pairwise["diff_shell_mae"].dropna().to_numpy(dtype=np.float64), bins=30)
    plt.xlabel("diff_shell_mae")
    plt.ylabel("count")
    plt.tight_layout()
    fig.savefig(out_dir / "diff_shell_mae_hist.png", dpi=200)
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(
        pairwise["n_samples_reference"].to_numpy(dtype=np.float64),
        pairwise["diff_shell_mae"].to_numpy(dtype=np.float64),
        s=16,
        alpha=0.7,
    )
    plt.xlabel("n_samples_reference")
    plt.ylabel("diff_shell_mae")
    plt.tight_layout()
    fig.savefig(out_dir / "diff_shell_mae_vs_cluster_size.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-level comparison for MuSRNet prediction CSVs")
    parser.add_argument("--pred", action="append", required=True, help="model_name=path/to/predictions.csv")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    model_paths = [parse_pred_arg(item) for item in args.pred]
    model_names = [model_name for model_name, _ in model_paths]
    if len(set(model_names)) != len(model_names):
        raise ValueError("Duplicate model names provided in --pred")
    if (args.reference is None) ^ (args.candidate is None):
        raise ValueError("--reference and --candidate must be provided together")
    if args.reference is not None and args.reference not in model_names:
        raise ValueError(f"Reference model '{args.reference}' not found in --pred arguments")
    if args.candidate is not None and args.candidate not in model_names:
        raise ValueError(f"Candidate model '{args.candidate}' not found in --pred arguments")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_tables = []
    total_samples_by_model: dict[str, int] = {}
    total_residues_by_model: dict[str, int] = {}
    for model_name, csv_path in model_paths:
        df = pd.read_csv(csv_path)
        df = normalize_columns(df)
        sample_metrics = compute_sample_metrics(df)
        cluster_metrics = compute_cluster_metrics(sample_metrics)
        cluster_metrics.insert(0, "model_name", model_name)
        cluster_tables.append(cluster_metrics)
        total_samples_by_model[model_name] = int(sample_metrics["sample_id"].nunique())
        total_residues_by_model[model_name] = int(sample_metrics["n_residues"].sum())

    cluster_metrics_wide = pd.concat(cluster_tables, ignore_index=True)
    cluster_metrics_long = make_long_table(cluster_metrics_wide)

    pairwise = None
    if args.reference is not None and args.candidate is not None:
        pairwise = make_pairwise_table(cluster_metrics_wide, args.reference, args.candidate)
        improved = pairwise[pairwise["diff_shell_mae"] < 0].copy()
        worsened = pairwise[pairwise["diff_shell_mae"] > 0].copy()
        improved = improved.sort_values("diff_shell_mae", ascending=True)
        worsened = worsened.sort_values("diff_shell_mae", ascending=False)
        pairwise.to_csv(out_dir / "pairwise_cluster_diff.csv", index=False)
        improved.to_csv(out_dir / "improved_clusters.csv", index=False)
        worsened.to_csv(out_dir / "worsened_clusters.csv", index=False)
        maybe_make_plots(pairwise, out_dir)
    else:
        pd.DataFrame().to_csv(out_dir / "pairwise_cluster_diff.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "improved_clusters.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "worsened_clusters.csv", index=False)

    cluster_metrics_long.to_csv(out_dir / "cluster_metrics_long.csv", index=False)
    cluster_metrics_wide.to_csv(out_dir / "cluster_metrics_wide.csv", index=False)

    summary = build_summary(cluster_metrics_wide, pairwise, args.reference, args.candidate)
    for model_name in summary["models"]:
        summary["model_summary"][model_name]["n_samples"] = total_samples_by_model[model_name]
        summary["model_summary"][model_name]["n_residues"] = total_residues_by_model[model_name]
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Models evaluated: {', '.join(summary['models'])}")
    if pairwise is not None:
        print(f"Common clusters: {len(pairwise)}")
    print("\nCluster-mean shell_mae:")
    for model_name in summary["models"]:
        value = summary["model_summary"][model_name]["shell_mae_cluster_mean"]
        print(f"  {model_name}: {value:.4f}" if not math.isnan(value) else f"  {model_name}: nan")
    if pairwise is not None:
        pairwise_summary = summary["pairwise_summary"]
        print("\nPairwise shell_mae:")
        print(f"  improved clusters: {pairwise_summary['n_clusters_shell_mae_improved']}")
        print(f"  worsened clusters: {pairwise_summary['n_clusters_shell_mae_worsened']}")
        print(f"  tied clusters: {pairwise_summary['n_clusters_shell_mae_tied']}")
        print(f"  mean diff: {pairwise_summary['mean_diff_shell_mae']:+.4f}")
        print(f"  median diff: {pairwise_summary['median_diff_shell_mae']:+.4f}")
    print("\nCluster-mean perturbed_auprc:")
    for model_name in summary["models"]:
        value = summary["model_summary"][model_name]["perturbed_auprc_cluster_mean"]
        print(f"  {model_name}: {value:.4f}" if not math.isnan(value) else f"  {model_name}: nan")
    print(f"\nOutputs written to:\n  {out_dir}")


if __name__ == "__main__":
    main()
