from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

SHELL_IDS = (0, 1, 2, 3, 4)
MEAN_CLUSTER_METRICS = (
    "global_mae",
    "shell_mae",
    "mae_shell_0",
    "mae_shell_1",
    "mae_shell_2",
    "mae_shell_3",
    "mae_shell_4",
    "perturbed_auroc",
    "perturbed_auprc",
    "derived_radius_mae",
    "radius_mae",
    "class_correct",
    "non_local_recall",
)


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size > 1 else float("nan")


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score)) if np.unique(y_true).size > 1 else float("nan")


def finite_mean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def first_numeric(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else float("nan")


def derive_sample_outcomes(
    df: pd.DataFrame,
    response_threshold: float,
    displacement_threshold: float,
    radius_threshold: float,
) -> dict[str, float]:
    pred_disp = df["pred_displacement"].to_numpy(dtype=np.float64)
    true_disp = df["true_displacement"].to_numpy(dtype=np.float64)
    pred_prob = df["pred_perturbed_prob"].to_numpy(dtype=np.float64)
    radii = df["radii"].to_numpy(dtype=np.float64) if "radii" in df.columns else None

    true_radius = first_numeric(df, "true_radius")
    if not np.isfinite(true_radius):
        if radii is None:
            true_radius = float("nan")
        else:
            mask = true_disp > displacement_threshold
            true_radius = float(radii[mask].max()) if mask.any() else 0.0

    pred_radius = first_numeric(df, "pred_radius")
    if not np.isfinite(pred_radius):
        if radii is None:
            pred_radius = float("nan")
        else:
            mask = pred_prob * pred_disp > response_threshold
            pred_radius = float(radii[mask].max()) if mask.any() else 0.0

    true_class_value = first_numeric(df, "true_class")
    if np.isfinite(true_class_value):
        true_class = int(true_class_value)
    elif np.isfinite(true_radius):
        max_true = float(true_disp.max()) if true_disp.size else 0.0
        true_class = 0 if max_true <= displacement_threshold else (1 if true_radius <= radius_threshold else 2)
    else:
        true_class = -1

    pred_class_value = first_numeric(df, "pred_class")
    if np.isfinite(pred_class_value):
        pred_class = int(pred_class_value)
    elif np.isfinite(pred_radius):
        max_pred = float(pred_disp.max()) if pred_disp.size else 0.0
        pred_class = 0 if max_pred <= displacement_threshold else (1 if pred_radius <= radius_threshold else 2)
    else:
        pred_class = -1

    radius_mae = abs(pred_radius - true_radius) if np.isfinite(pred_radius) and np.isfinite(true_radius) else float("nan")
    class_valid = true_class >= 0 and pred_class >= 0
    return {
        "true_radius": true_radius,
        "pred_radius": pred_radius,
        "derived_radius_mae": radius_mae,
        "radius_mae": radius_mae,
        "true_class": float(true_class) if true_class >= 0 else float("nan"),
        "pred_class": float(pred_class) if pred_class >= 0 else float("nan"),
        "class_correct": float(true_class == pred_class) if class_valid else float("nan"),
        "non_local_recall": float(pred_class == 2) if true_class == 2 and pred_class >= 0 else float("nan"),
    }


def compute_sample_metrics(
    df: pd.DataFrame,
    response_threshold: float = 0.5,
    displacement_threshold: float = 1.0,
    radius_threshold: float = 8.0,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    required = {
        "sample_id",
        "cluster_id_30",
        "shell_id",
        "true_displacement",
        "pred_displacement",
        "true_perturbed",
        "pred_perturbed_prob",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    warnings = warnings if warnings is not None else []
    rows: list[dict[str, Any]] = []

    grouped = df.groupby(["cluster_id_30", "sample_id"], sort=False, observed=True)
    for (cluster_id, sample_id), sample_df in tqdm(grouped, total=grouped.ngroups, desc="Samples", leave=False):
        errors = np.abs(
            sample_df["pred_displacement"].to_numpy(dtype=np.float64)
            - sample_df["true_displacement"].to_numpy(dtype=np.float64)
        )
        shells = sample_df["shell_id"].to_numpy(dtype=np.int64)
        row: dict[str, Any] = {
            "sample_id": str(sample_id),
            "cluster_id_30": str(cluster_id),
            "global_mae": float(errors.mean()),
            "n_residues": int(len(sample_df)),
        }

        shell_values = []
        for shell_idx in SHELL_IDS:
            mask = shells == shell_idx
            value = float(errors[mask].mean()) if mask.any() else float("nan")
            row[f"mae_shell_{shell_idx}"] = value
            if np.isfinite(value):
                shell_values.append(value)
        row["shell_mae"] = float(np.mean(shell_values)) if shell_values else float("nan")

        y_true = sample_df["true_perturbed"].to_numpy(dtype=np.int64)
        y_score = sample_df["pred_perturbed_prob"].to_numpy(dtype=np.float64)
        row["perturbed_auroc"] = safe_auroc(y_true, y_score)
        row["perturbed_auprc"] = safe_auprc(y_true, y_score)
        if np.unique(y_true).size < 2:
            warnings.append(f"AUROC/AUPRC undefined for one-class sample {sample_id}")

        row.update(
            derive_sample_outcomes(
                sample_df,
                response_threshold=response_threshold,
                displacement_threshold=displacement_threshold,
                radius_threshold=radius_threshold,
            )
        )
        rows.append(row)

    return pd.DataFrame(rows)


def compute_cluster_metrics(sample_metrics_df: pd.DataFrame) -> pd.DataFrame:
    mean_columns = [column for column in MEAN_CLUSTER_METRICS if column in sample_metrics_df.columns]
    grouped = sample_metrics_df.groupby("cluster_id_30", sort=False, observed=True)

    cluster_df = grouped[mean_columns].mean().reset_index()
    counts = grouped.agg(n_samples=("sample_id", "nunique"), n_residues=("n_residues", "sum")).reset_index()
    cluster_df = cluster_df.merge(counts, on="cluster_id_30", how="left")

    class_rows = []
    for cluster_id, group_df in grouped:
        valid = group_df[["true_class", "pred_class"]].dropna()
        if valid.empty:
            macro_f1 = non_local_f1 = float("nan")
        else:
            y_true = valid["true_class"].to_numpy(dtype=np.int64)
            y_pred = valid["pred_class"].to_numpy(dtype=np.int64)
            macro_f1 = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0))
            non_local_f1 = float(f1_score(y_true == 2, y_pred == 2, zero_division=0))
        class_rows.append(
            {
                "cluster_id_30": str(cluster_id),
                "derived_class_macro_f1": macro_f1,
                "class_macro_f1": macro_f1,
                "non_local_f1": non_local_f1,
            }
        )

    return cluster_df.merge(pd.DataFrame(class_rows), on="cluster_id_30", how="left")


def compute_overall_metrics(
    df: pd.DataFrame,
    sample_metrics_df: pd.DataFrame,
    cluster_metrics_df: pd.DataFrame,
) -> dict[str, float]:
    errors = np.abs(
        df["pred_displacement"].to_numpy(dtype=np.float64)
        - df["true_displacement"].to_numpy(dtype=np.float64)
    )
    shells = df["shell_id"].to_numpy(dtype=np.int64)
    summary: dict[str, float] = {"global_mae": float(errors.mean())}

    shell_values = []
    for shell_idx in SHELL_IDS:
        mask = shells == shell_idx
        value = float(errors[mask].mean()) if mask.any() else float("nan")
        summary[f"mae_shell_{shell_idx}"] = value
        if np.isfinite(value):
            shell_values.append(value)
    summary["shell_mae"] = float(np.mean(shell_values)) if shell_values else float("nan")

    y_true = df["true_perturbed"].to_numpy(dtype=np.int64)
    y_score = df["pred_perturbed_prob"].to_numpy(dtype=np.float64)
    summary["perturbed_auroc"] = safe_auroc(y_true, y_score)
    summary["perturbed_auprc"] = safe_auprc(y_true, y_score)

    summary["derived_radius_mae"] = finite_mean(sample_metrics_df["derived_radius_mae"])
    summary["radius_mae"] = summary["derived_radius_mae"]

    valid = sample_metrics_df[["true_class", "pred_class"]].dropna()
    if valid.empty:
        macro_f1 = non_local_f1 = float("nan")
    else:
        class_true = valid["true_class"].to_numpy(dtype=np.int64)
        class_pred = valid["pred_class"].to_numpy(dtype=np.int64)
        macro_f1 = float(f1_score(class_true, class_pred, labels=[0, 1, 2], average="macro", zero_division=0))
        non_local_f1 = float(f1_score(class_true == 2, class_pred == 2, zero_division=0))

    summary["derived_class_macro_f1"] = macro_f1
    summary["class_macro_f1"] = macro_f1
    summary["non_local_f1"] = non_local_f1

    cluster_map = {
        "cluster_avg_global_mae": "global_mae",
        "cluster_avg_shell_mae": "shell_mae",
        "cluster_avg_auroc": "perturbed_auroc",
        "cluster_avg_auprc": "perturbed_auprc",
        "cluster_avg_radius_mae": "derived_radius_mae",
        "cluster_avg_class_macro_f1": "derived_class_macro_f1",
    }
    for output_name, column in cluster_map.items():
        summary[output_name] = finite_mean(cluster_metrics_df[column])

    summary["n_samples"] = int(sample_metrics_df["sample_id"].nunique())
    summary["n_clusters"] = int(cluster_metrics_df["cluster_id_30"].nunique())
    summary["n_residues"] = int(len(df))
    return summary


def summarize_predictions(
    df: pd.DataFrame,
    response_threshold: float = 0.5,
    displacement_threshold: float = 1.0,
    radius_threshold: float = 8.0,
    warnings: list[str] | None = None,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    sample_metrics_df = compute_sample_metrics(
        df,
        response_threshold=response_threshold,
        displacement_threshold=displacement_threshold,
        radius_threshold=radius_threshold,
        warnings=warnings,
    )
    cluster_metrics_df = compute_cluster_metrics(sample_metrics_df)
    summary = compute_overall_metrics(df, sample_metrics_df, cluster_metrics_df)
    return summary, sample_metrics_df, cluster_metrics_df


def compare_cluster_metric(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    metric: str,
    higher_is_better: bool,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    merged = candidate[["cluster_id_30", metric]].merge(
        baseline[["cluster_id_30", metric]],
        on="cluster_id_30",
        how="inner",
        suffixes=("_candidate", "_baseline"),
    )
    candidate_values = merged[f"{metric}_candidate"].to_numpy(dtype=np.float64)
    baseline_values = merged[f"{metric}_baseline"].to_numpy(dtype=np.float64)
    valid = np.isfinite(candidate_values) & np.isfinite(baseline_values)
    diff = candidate_values[valid] - baseline_values[valid]

    improved = diff > 0 if higher_is_better else diff < 0
    worsened = diff < 0 if higher_is_better else diff > 0

    result = {
        "n_common_clusters": int(diff.size),
        "n_improved_clusters": int(improved.sum()),
        "n_worsened_clusters": int(worsened.sum()),
        "fraction_improved": float(improved.mean()) if diff.size else float("nan"),
        "mean_diff": float(diff.mean()) if diff.size else float("nan"),
        "median_diff": float(np.median(diff)) if diff.size else float("nan"),
        "bootstrap_95ci_low": float("nan"),
        "bootstrap_95ci_high": float("nan"),
        "wilcoxon_pvalue": float("nan"),
    }

    if diff.size:
        rng = np.random.default_rng(seed)
        bootstrap = np.empty(n_bootstrap, dtype=np.float64)
        for i in tqdm(range(n_bootstrap), desc=f"Bootstrap {metric}", leave=False):
            bootstrap[i] = diff[rng.integers(0, diff.size, diff.size)].mean()
        result["bootstrap_95ci_low"], result["bootstrap_95ci_high"] = map(
            float, np.percentile(bootstrap, [2.5, 97.5])
        )

    if diff.size >= 2 and not np.allclose(diff, 0.0):
        result["wilcoxon_pvalue"] = float(wilcoxon(diff).pvalue)

    return result