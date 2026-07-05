from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.dataset import load_samples_manifest
from musrnet.train_utils import load_yaml


class MissingMetadataError(ValueError):
    def __init__(self, message: str, missing_df: pd.DataFrame) -> None:
        super().__init__(message)
        self.missing_df = missing_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cluster-safe time split for MuSRNet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cluster-policy", required=True, choices=["latest_release"])
    parser.add_argument("--train-max-year", type=int, required=True)
    parser.add_argument("--valid-year", type=int, required=True)
    parser.add_argument("--test-min-year", type=int, required=True)
    return parser.parse_args()


def assign_cluster_split(cluster_year: int, train_max_year: int, valid_year: int, test_min_year: int) -> str:
    if cluster_year <= train_max_year:
        return "train"
    if cluster_year == valid_year:
        return "valid"
    if cluster_year >= test_min_year:
        return "test"
    raise ValueError(
        f"Cluster year {cluster_year} does not match any split rule: "
        f"train<= {train_max_year}, valid== {valid_year}, test>= {test_min_year}"
    )


def build_time_split(
    manifest_sample_ids: list[str],
    sample_df: pd.DataFrame,
    *,
    train_max_year: int,
    valid_year: int,
    test_min_year: int,
) -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    df = sample_df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["cluster_id_30"] = df["cluster_id_30"].astype(str)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    manifest_df = pd.DataFrame({"sample_id": sorted(set(str(sample_id) for sample_id in manifest_sample_ids))})
    merged = manifest_df.merge(df[["sample_id", "cluster_id_30", "release_date"]], on="sample_id", how="left")

    missing_mask = merged["cluster_id_30"].isna() | merged["release_date"].isna()
    if missing_mask.any():
        raise MissingMetadataError(
            "Missing required metadata for some manifest samples",
            merged.loc[missing_mask].copy(),
        )

    merged["sample_year"] = merged["release_date"].dt.year.astype(int)

    cluster_df = (
        merged.groupby("cluster_id_30", as_index=False)
        .agg(
            cluster_year=("sample_year", "max"),
            n_samples=("sample_id", "count"),
            min_sample_year=("sample_year", "min"),
            max_sample_year=("sample_year", "max"),
        )
    )
    cluster_df["split"] = cluster_df["cluster_year"].apply(
        lambda year: assign_cluster_split(year, train_max_year, valid_year, test_min_year)
    )

    merged = merged.merge(cluster_df[["cluster_id_30", "cluster_year", "split"]], on="cluster_id_30", how="left")
    merged = merged.sort_values(["split", "cluster_year", "sample_year", "sample_id"]).reset_index(drop=True)
    merged["release_date"] = merged["release_date"].dt.strftime("%Y-%m-%d")

    splits = {
        split: sorted(merged.loc[merged["split"] == split, "sample_id"].astype(str).tolist())
        for split in ["train", "valid", "test"]
    }
    empty_splits = [split for split, sample_ids in splits.items() if not sample_ids]
    if empty_splits:
        raise ValueError(f"Time split has empty split(s): {empty_splits}")

    cluster_sets = {
        split: set(cluster_df.loc[cluster_df["split"] == split, "cluster_id_30"].astype(str).tolist())
        for split in ["train", "valid", "test"]
    }
    overlap = {
        "train_valid": sorted(cluster_sets["train"] & cluster_sets["valid"]),
        "train_test": sorted(cluster_sets["train"] & cluster_sets["test"]),
        "valid_test": sorted(cluster_sets["valid"] & cluster_sets["test"]),
    }
    assert all(len(values) == 0 for values in overlap.values()), f"Cluster overlap detected: {overlap}"

    def year_range(frame: pd.DataFrame, column: str) -> list[int | None]:
        if frame.empty:
            return [None, None]
        return [int(frame[column].min()), int(frame[column].max())]

    summary = {
        "split_rule": {
            "train": f"cluster_year <= {train_max_year}",
            "valid": f"cluster_year == {valid_year}",
            "test": f"cluster_year >= {test_min_year}",
            "cluster_policy": "latest_release",
        },
        "n_samples": {split: len(splits[split]) for split in ["train", "valid", "test"]},
        "n_clusters": {
            split: int(cluster_df.loc[cluster_df["split"] == split, "cluster_id_30"].nunique())
            for split in ["train", "valid", "test"]
        },
        "sample_year_range": {
            split: year_range(merged.loc[merged["split"] == split], "sample_year")
            for split in ["train", "valid", "test"]
        },
        "cluster_year_range": {
            split: year_range(cluster_df.loc[cluster_df["split"] == split], "cluster_year")
            for split in ["train", "valid", "test"]
        },
        "cluster_overlap": overlap,
    }

    sample_audit = merged[
        ["sample_id", "cluster_id_30", "release_date", "sample_year", "cluster_year", "split"]
    ].copy()
    cluster_audit = cluster_df[
        ["cluster_id_30", "cluster_year", "split", "n_samples", "min_sample_year", "max_sample_year"]
    ].sort_values(["split", "cluster_year", "cluster_id_30"]).reset_index(drop=True)
    return splits, sample_audit, cluster_audit, summary


def audit_dir_from_out(out_path: Path) -> Path:
    stem = out_path.stem
    if stem.startswith("splits_time_"):
        stem = stem.replace("splits_time_", "time_split_", 1)
    elif stem.startswith("splits_"):
        stem = stem.replace("splits_", "", 1)
    return out_path.parent / f"{stem}_audit"


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing split file: {out_path}")

    config = load_yaml(args.config)
    manifest = load_samples_manifest(config["paths"]["samples"])
    manifest_sample_ids = [str(sample_id) for sample_id in manifest["sample_ids"]]

    sample_df = pd.read_csv(args.csv)
    required = ["sample_id", "cluster_id_30", "release_date"]
    missing_columns = [column for column in required if column not in sample_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {missing_columns}")

    try:
        splits, sample_audit, cluster_audit, summary = build_time_split(
            manifest_sample_ids,
            sample_df,
            train_max_year=args.train_max_year,
            valid_year=args.valid_year,
            test_min_year=args.test_min_year,
        )
    except MissingMetadataError as exc:
        missing_path = out_path.parent / "time_split_missing_metadata.csv"
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        exc.missing_df.to_csv(missing_path, index=False)
        raise ValueError(f"{exc}. Details written to {missing_path}") from None

    audit_dir = audit_dir_from_out(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=False)

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)
    sample_audit.to_csv(audit_dir / "time_split_samples.csv", index=False)
    cluster_audit.to_csv(audit_dir / "time_split_clusters.csv", index=False)
    with (audit_dir / "time_split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Time split created:")
    for split in ["train", "valid", "test"]:
        print(f"{split}: {summary['n_samples'][split]} samples, {summary['n_clusters'][split]} clusters")
    print(f"Split JSON: {out_path}")
    print(f"Audit dir: {audit_dir}")


if __name__ == "__main__":
    main()
