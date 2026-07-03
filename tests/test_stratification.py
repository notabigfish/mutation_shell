from __future__ import annotations

import math

import pandas as pd

from musrnet.stratification import (
    assign_charge_group,
    assign_exposure_group,
    assign_glypro_group,
    assign_secondary_structure_group,
    assign_substitution_size_group,
    compute_pairwise_stratified_diffs,
    compute_sample_metrics,
    compute_stratified_metrics,
)


def test_assign_substitution_size_group() -> None:
    assert assign_substitution_size_group("A", "W") == "small_to_large"
    assert assign_substitution_size_group("W", "A") == "large_to_small"
    assert assign_substitution_size_group("L", "I") == "size_neutral"


def test_assign_charge_group() -> None:
    assert assign_charge_group("D", "N") == "charged_to_neutral"
    assert assign_charge_group("A", "K") == "neutral_to_charged"
    assert assign_charge_group("D", "K") == "negative_to_positive"
    assert assign_charge_group("K", "R") == "charged_to_charged_same_sign"
    assert assign_charge_group("A", "V") == "neutral_to_neutral"


def test_assign_glypro_group() -> None:
    assert assign_glypro_group("A", "G") == "gly_or_pro_involved"
    assert assign_glypro_group("P", "A") == "gly_or_pro_involved"
    assert assign_glypro_group("A", "V") == "no_gly_or_pro"


def test_assign_secondary_structure_group() -> None:
    for code in ["H", "G", "I"]:
        assert assign_secondary_structure_group(code) == "helix"
    for code in ["E", "B"]:
        assert assign_secondary_structure_group(code) == "sheet"
    for code in ["T", "S", "-"]:
        assert assign_secondary_structure_group(code) == "loop"
    assert assign_secondary_structure_group("unknown") == "unknown"


def test_assign_exposure_group() -> None:
    assert assign_exposure_group(0.10, 0.20) == "buried"
    assert assign_exposure_group(0.25, 0.20) == "exposed"
    assert assign_exposure_group(float("nan"), 0.20) == "unknown"


def test_compute_sample_metrics() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s1"] * 5 + ["s2"] * 5,
            "cluster_id_30": ["c1"] * 5 + ["c2"] * 5,
            "shell_id": [0, 1, 2, 3, 4] * 2,
            "true_displacement": [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.5, 1.5, 2.5, 3.5],
            "pred_displacement": [0.0, 1.5, 1.0, 4.0, 4.5, 0.0, 0.0, 2.0, 2.0, 3.0],
            "true_perturbed": [0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
            "pred_perturbed_prob": [0.1, 0.2, 0.8, 0.9, 0.7, 0.1, 0.3, 0.8, 0.7, 0.6],
            "radii": [2.0, 6.0, 10.0, 14.0, 20.0] * 2,
        }
    )
    metrics = compute_sample_metrics(df)
    row = metrics.set_index("sample_id").loc["s1"]
    assert math.isclose(row["global_mae"], 0.6)
    assert math.isclose(row["shell_mae"], 0.6)
    assert math.isclose(row["mae_shell_0"], 0.0)


def test_cluster_mean_equal_weighting() -> None:
    df = pd.DataFrame(
        {
            "model_name": ["m"] * 4,
            "sample_id": ["a", "b", "c", "d"],
            "cluster_id_30": ["c1", "c1", "c1", "c2"],
            "n_residues": [10, 10, 10, 10],
            "shell_mae": [1.0, 1.0, 1.0, 3.0],
            "exposure_group": ["buried"] * 4,
        }
    )
    long_df, _ = compute_stratified_metrics(
        sample_metrics_df=df,
        factor_columns=["exposure_group"],
        metric_columns=["shell_mae"],
        bootstrap_iters=100,
        seed=42,
        warnings=[],
    )
    row = long_df.iloc[0]
    assert math.isclose(row["sample_mean"], 1.5)
    assert math.isclose(row["cluster_mean"], 2.0)


def test_pairwise_diff_direction_logic() -> None:
    cluster_df = pd.DataFrame(
        {
            "model_name": ["ref", "cand", "ref", "cand", "ref", "cand", "ref", "cand", "ref", "cand"] * 2,
            "factor": ["grouping"] * 20,
            "group": ["g1"] * 20,
            "metric": ["shell_mae"] * 10 + ["perturbed_auprc"] * 10,
            "cluster_id_30": ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4", "c5", "c5"] * 2,
            "value": [2.0, 1.0, 2.5, 1.5, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 0.2, 0.4, 0.3, 0.5, 0.4, 0.6, 0.5, 0.7, 0.6, 0.8],
            "n_samples": [1] * 20,
            "n_residues": [10] * 20,
        }
    )
    diff_df = compute_pairwise_stratified_diffs(
        cluster_metric_df=cluster_df,
        reference="ref",
        candidate="cand",
        bootstrap_iters=100,
        seed=42,
        warnings=[],
    )
    shell_row = diff_df[diff_df["metric"] == "shell_mae"].iloc[0]
    auprc_row = diff_df[diff_df["metric"] == "perturbed_auprc"].iloc[0]
    assert shell_row["n_clusters_improved"] == 5
    assert auprc_row["n_clusters_improved"] == 5
