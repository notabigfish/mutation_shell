from __future__ import annotations

import pandas as pd
import pytest

from scripts.create_time_split import build_time_split


def make_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_time_split_no_cluster_overlap() -> None:
    manifest_ids = ["s1", "s2", "s3", "s4"]
    df = make_df(
        [
            {"sample_id": "s1", "cluster_id_30": "A", "release_date": "2022-01-01"},
            {"sample_id": "s2", "cluster_id_30": "B", "release_date": "2023-02-01"},
            {"sample_id": "s3", "cluster_id_30": "C", "release_date": "2024-03-01"},
            {"sample_id": "s4", "cluster_id_30": "D", "release_date": "2021-04-01"},
        ]
    )
    _, _, cluster_audit, _ = build_time_split(
        manifest_ids,
        df,
        train_max_year=2022,
        valid_year=2023,
        test_min_year=2024,
    )
    train_clusters = set(cluster_audit.loc[cluster_audit["split"] == "train", "cluster_id_30"])
    valid_clusters = set(cluster_audit.loc[cluster_audit["split"] == "valid", "cluster_id_30"])
    test_clusters = set(cluster_audit.loc[cluster_audit["split"] == "test", "cluster_id_30"])
    assert train_clusters.isdisjoint(valid_clusters)
    assert train_clusters.isdisjoint(test_clusters)
    assert valid_clusters.isdisjoint(test_clusters)


def test_latest_release_cluster_policy() -> None:
    manifest_ids = ["s1", "s2", "s3"]
    df = make_df(
        [
            {"sample_id": "s1", "cluster_id_30": "A", "release_date": "2021-01-01"},
            {"sample_id": "s2", "cluster_id_30": "A", "release_date": "2024-01-01"},
            {"sample_id": "s3", "cluster_id_30": "B", "release_date": "2023-01-01"},
        ]
    )
    splits, _, cluster_audit, _ = build_time_split(
        manifest_ids,
        df,
        train_max_year=2022,
        valid_year=2023,
        test_min_year=2024,
    )
    assert "s1" in splits["test"]
    assert "s2" in splits["test"]
    assert cluster_audit.set_index("cluster_id_30").loc["A", "split"] == "test"


def test_year_assignment() -> None:
    manifest_ids = ["s1", "s2", "s3"]
    df = make_df(
        [
            {"sample_id": "s1", "cluster_id_30": "A", "release_date": "2022-01-01"},
            {"sample_id": "s2", "cluster_id_30": "B", "release_date": "2023-01-01"},
            {"sample_id": "s3", "cluster_id_30": "C", "release_date": "2024-01-01"},
        ]
    )
    splits, _, _, _ = build_time_split(
        manifest_ids,
        df,
        train_max_year=2022,
        valid_year=2023,
        test_min_year=2024,
    )
    assert splits["train"] == ["s1"]
    assert splits["valid"] == ["s2"]
    assert splits["test"] == ["s3"]


def test_missing_release_date_fails() -> None:
    manifest_ids = ["s1", "s2", "s3"]
    df = make_df(
        [
            {"sample_id": "s1", "cluster_id_30": "A", "release_date": None},
            {"sample_id": "s2", "cluster_id_30": "B", "release_date": "2023-01-01"},
            {"sample_id": "s3", "cluster_id_30": "C", "release_date": "2024-01-01"},
        ]
    )
    with pytest.raises(ValueError):
        build_time_split(
            manifest_ids,
            df,
            train_max_year=2022,
            valid_year=2023,
            test_min_year=2024,
        )


def test_empty_split_fails() -> None:
    manifest_ids = ["s1", "s2"]
    df = make_df(
        [
            {"sample_id": "s1", "cluster_id_30": "A", "release_date": "2022-01-01"},
            {"sample_id": "s2", "cluster_id_30": "B", "release_date": "2023-01-01"},
        ]
    )
    with pytest.raises(ValueError, match="empty split"):
        build_time_split(
            manifest_ids,
            df,
            train_max_year=2022,
            valid_year=2023,
            test_min_year=2024,
        )
