from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def _safe_metric(fn, *args):
    try:
        return float(fn(*args))
    except Exception:
        return float("nan")


def _nanmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def compute_metrics(records: dict[str, Any]) -> dict[str, float]:
    true_disp = np.asarray(records["true_disp"], dtype=np.float32)
    pred_disp = np.asarray(records["pred_disp"], dtype=np.float32)
    shell_id = np.asarray(records["shell_id"], dtype=np.int64)
    true_pert = np.asarray(records["true_perturbed"], dtype=np.float32)
    pred_pert = np.asarray(records["pred_perturbed_prob"], dtype=np.float32)
    true_radius = np.asarray(records["true_radius"], dtype=np.float32)
    pred_radius = np.asarray(records["pred_radius"], dtype=np.float32)
    true_class = np.asarray(records["true_class"], dtype=np.int64)
    pred_class = np.asarray(records["pred_class"], dtype=np.int64)
    cluster_ids = np.asarray(records["cluster_id_30"])

    abs_error = np.abs(pred_disp - true_disp)
    radius_mae = float(np.abs(pred_radius - true_radius).mean()) if true_radius.size else float("nan")
    class_macro_f1 = _safe_metric(lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), true_class, pred_class)
    metrics = {
        "global_mae": float(abs_error.mean()) if abs_error.size else float("nan"),
        "perturbed_auroc": _safe_metric(roc_auc_score, true_pert, pred_pert),
        "perturbed_auprc": _safe_metric(average_precision_score, true_pert, pred_pert),
        "radius_mae": radius_mae,
        "derived_radius_mae": radius_mae,
        "class_macro_f1": class_macro_f1,
        "derived_class_macro_f1": class_macro_f1,
    }

    shell_maes = []
    for shell_idx in range(5):
        mask = shell_id == shell_idx
        value = float(abs_error[mask].mean()) if mask.any() else float("nan")
        metrics[f"mae_shell_{shell_idx}"] = value
        if mask.any():
            shell_maes.append(value)
    metrics["shell_mae"] = float(np.mean(shell_maes)) if shell_maes else float("nan")

    cluster_shell_values = []
    cluster_global_values = []
    cluster_auprc_values = []
    cluster_radius_values = []
    cluster_class_values = []
    for cluster_id in np.unique(cluster_ids):
        cluster_mask = cluster_ids == cluster_id
        cluster_global_values.append(float(abs_error[cluster_mask].mean()))
        cluster_radius_values.append(float(np.abs(pred_radius[cluster_mask] - true_radius[cluster_mask]).mean()))
        cluster_class_values.append(
            _safe_metric(lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), true_class[cluster_mask], pred_class[cluster_mask])
        )
        if len(np.unique(true_pert[cluster_mask])) > 1:
            cluster_auprc_values.append(_safe_metric(average_precision_score, true_pert[cluster_mask], pred_pert[cluster_mask]))
        per_shell = []
        for shell_idx in range(5):
            shell_mask = cluster_mask & (shell_id == shell_idx)
            if shell_mask.any():
                per_shell.append(float(abs_error[shell_mask].mean()))
        if per_shell:
            cluster_shell_values.append(float(np.mean(per_shell)))

    metrics["cluster_avg_shell_mae"] = _nanmean(cluster_shell_values)
    metrics["cluster_avg_global_mae"] = _nanmean(cluster_global_values)
    metrics["cluster_avg_auprc"] = _nanmean(cluster_auprc_values)
    metrics["cluster_avg_radius_mae"] = _nanmean(cluster_radius_values)
    metrics["cluster_avg_class_macro_f1"] = _nanmean(cluster_class_values)
    return metrics
