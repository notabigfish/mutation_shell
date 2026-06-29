from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

def parse_pred_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Expected --pred model_name=path/to/predictions_test.csv")
    name, path = value.split("=", 1)
    return name, Path(path)


def derive_per_sample(
    sample_df: pd.DataFrame,
    response_threshold: float,
    radius_threshold: float,
    displacement_threshold: float,
) -> dict:
    pred_disp = sample_df["pred_displacement"].to_numpy(dtype=np.float64)
    pred_prob = sample_df["pred_perturbed_prob"].to_numpy(dtype=np.float64)
    true_disp = sample_df["true_displacement"].to_numpy(dtype=np.float64)
    radii = sample_df["radii"].to_numpy(dtype=np.float64)

    true_radius = float(sample_df["true_radius"].iloc[0])

    score = pred_prob * pred_disp
    pred_mask = score > response_threshold

    if pred_mask.any():
        pred_radius = float(radii[pred_mask].max())
    else:
        pred_radius = 0.0

    max_pred_disp = float(pred_disp.max()) if pred_disp.size else 0.0
    max_true_disp = float(true_disp.max()) if true_disp.size else 0.0

    if max_pred_disp <= displacement_threshold:
        pred_class = 0
    elif pred_radius <= radius_threshold:
        pred_class = 1
    else:
        pred_class = 2

    if max_true_disp <= displacement_threshold:
        true_class = 0
    elif true_radius <= radius_threshold:
        true_class = 1
    else:
        true_class = 2

    return {
        "sample_id": str(sample_df["sample_id"].iloc[0]),
        "cluster_id_30": str(sample_df["cluster_id_30"].iloc[0]),
        "true_radius": true_radius,
        "pred_radius": pred_radius,
        "radius_abs_error": abs(pred_radius - true_radius),
        "true_class": true_class,
        "pred_class": pred_class,
    }


def evaluate_threshold_pair(
    df: pd.DataFrame,
    response_threshold: float,
    radius_threshold: float,
    displacement_threshold: float,
) -> dict:
    rows = []

    for _, sample_df in df.groupby("sample_id", sort=False):
        rows.append(
            derive_per_sample(
                sample_df=sample_df,
                response_threshold=response_threshold,
                radius_threshold=radius_threshold,
                displacement_threshold=displacement_threshold,
            )
        )

    sample_metrics = pd.DataFrame(rows)

    derived_radius_mae = float(sample_metrics["radius_abs_error"].mean())

    derived_class_macro_f1 = float(
        f1_score(
            sample_metrics["true_class"],
            sample_metrics["pred_class"],
            average="macro",
            labels=[0, 1, 2],
            zero_division=0,
        )
    )

    cluster_rows = []
    for cluster_id, cluster_df in tqdm(sample_metrics.groupby("cluster_id_30", sort=False), total=sample_metrics["cluster_id_30"].nunique(), desc="Evaluating clusters"):
        cluster_radius_mae = float(cluster_df["radius_abs_error"].mean())
        cluster_class_macro_f1 = float(
            f1_score(
                cluster_df["true_class"],
                cluster_df["pred_class"],
                average="macro",
                labels=[0, 1, 2],
                zero_division=0,
            )
        )

        cluster_rows.append(
            {
                "cluster_id_30": cluster_id,
                "cluster_radius_mae": cluster_radius_mae,
                "cluster_class_macro_f1": cluster_class_macro_f1,
            }
        )

    cluster_metrics = pd.DataFrame(cluster_rows)

    return {
        "derived_radius_mae": derived_radius_mae,
        "derived_class_macro_f1": derived_class_macro_f1,
        "cluster_avg_radius_mae": float(cluster_metrics["cluster_radius_mae"].mean()),
        "cluster_avg_class_macro_f1": float(cluster_metrics["cluster_class_macro_f1"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--response-thresholds", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.7])
    parser.add_argument("--radius-thresholds", nargs="+", type=float, default=[6.0, 8.0, 10.0, 12.0])
    parser.add_argument("--displacement-thresholds", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    args = parser.parse_args()

    pred_items = [parse_pred_arg(x) for x in args.pred]
    all_rows = []

    for model_name, path in pred_items:
        df = pd.read_csv(path)

        required = [
            "sample_id",
            "cluster_id_30",
            "radii",
            "true_displacement",
            "pred_displacement",
            "pred_perturbed_prob",
            "true_radius",
        ]
        missing = [x for x in required if x not in df.columns]
        if missing:
            raise ValueError(f"{model_name} missing columns: {missing}")
        num_exps = len(args.response_thresholds) * len(args.radius_thresholds) * len(args.displacement_thresholds)
        ind_exp = 0
        for dp in args.displacement_thresholds:
            for eta in args.response_thresholds:
                for rc in args.radius_thresholds:
                    print(f"{ind_exp+1} / {num_exps}")
                    metrics = evaluate_threshold_pair(
                        df=df,
                        response_threshold=eta,
                        radius_threshold=rc,
                        displacement_threshold=dp,
                    )

                    all_rows.append(
                        {
                            "model_name": model_name,
                            "response_threshold": eta,
                            "radius_threshold": rc,
                            "displacement_threshold": dp,
                            **metrics,
                        }
                    )
                    ind_exp += 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()