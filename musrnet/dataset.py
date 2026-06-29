from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from musrnet.constants import AA_TO_INDEX, AMINO_ACIDS, RBF_CENTERS, RBF_SIGMA
from musrnet.esm_embed import load_chain_esm, open_esm_lmdb
from musrnet.graph import build_graph
from tqdm import tqdm

def load_samples_manifest(samples_path: str | Path) -> dict[str, Any]:
    samples_path = Path(samples_path)
    if samples_path.suffix == ".json":
        with samples_path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if "sample_ids" not in obj or "samples_dir" not in obj:
            raise ValueError("Unsupported samples file format")
        return obj
    obj = torch.load(samples_path, map_location="cpu")
    if isinstance(obj, list):
        sample_dir = Path(samples_path).with_suffix("")
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_ids = []
        metadata = []
        for sample in tqdm(obj, desc="Saving samples (load_samples_manifest)"):
            sample_path = sample_dir / f"{sample['sample_id']}.pt"
            torch.save(sample, sample_path)
            sample_ids.append(sample["sample_id"])
            metadata.append(
                {
                    "sample_id": sample["sample_id"],
                    "cluster_id_30": sample["cluster_id_30"],
                    "length": int(sample["coords_wt"].shape[0]),
                }
            )
        manifest = {
            "format": "musrnet_manifest_v1",
            "samples_dir": str(sample_dir),
            "sample_ids": sample_ids,
            "metadata": metadata,
            "rejections": {},
        }
        torch.save(manifest, samples_path)
        return manifest
    if not isinstance(obj, dict):
        raise ValueError("Unsupported samples file format")
    if "sample_files" in obj and "sample_ids" not in obj:
        obj["sample_ids"] = [Path(path).stem for path in obj["sample_files"]]
    if "sample_ids" not in obj or "samples_dir" not in obj:
        raise ValueError("Unsupported samples file format")
    return obj


def get_sample_path(manifest: dict[str, Any], sample_id: str) -> Path:
    return Path(manifest["samples_dir"]) / f"{sample_id}.pt"


def load_sample_from_manifest(manifest: dict[str, Any], sample_id: str) -> dict[str, Any]:
    return torch.load(get_sample_path(manifest, sample_id), map_location="cpu")


def iter_sample_paths(manifest: dict[str, Any]) -> Iterable[Path]:
    for sample_id in manifest["sample_ids"]:
        yield get_sample_path(manifest, sample_id)


def one_hot_aa(aa: str) -> np.ndarray:
    vec = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    vec[AA_TO_INDEX[aa]] = 1.0
    return vec


def rbf_features(values: np.ndarray) -> np.ndarray:
    return np.exp(-((values[:, None] - RBF_CENTERS[None, :]) ** 2) / (2.0 * (RBF_SIGMA**2))).astype(np.float32)


def build_basic_features(sample: dict[str, Any]) -> torch.FloatTensor:
    wt_sequence = sample["wt_sequence"]
    mut_pos = int(sample["mut_pos"])
    mutation_vector = one_hot_aa(sample["mut_aa"]) - one_hot_aa(sample["wt_aa"])
    aa_features = np.stack([one_hot_aa(aa) for aa in wt_sequence], axis=0)
    mutation_site = np.zeros((len(wt_sequence), 1), dtype=np.float32)
    mutation_site[mut_pos, 0] = 1.0
    mutation_broadcast = np.repeat(mutation_vector[None, :], len(wt_sequence), axis=0)
    radial = rbf_features(sample["radii"].cpu().numpy())
    basic = np.concatenate([aa_features, mutation_site, mutation_broadcast, radial], axis=1)
    return torch.from_numpy(basic).float()


def _cluster_source_ids(cluster_data: Any) -> set[str]:
    source_ids: set[str] = set()
    if isinstance(cluster_data, dict):
        for key, values in cluster_data.items():
            source_ids.add(str(key))
            for value in values:
                source_ids.add(str(value))
    return source_ids


def create_or_load_splits(
    samples_manifest: dict[str, Any],
    splits_path: str | Path,
    cluster_pkl_path: str | Path,
    seed: int,
) -> dict[str, list[str]]:
    splits_path = Path(splits_path)
    manifest_sample_ids = set(samples_manifest["sample_ids"])
    if splits_path.exists():
        with splits_path.open("r", encoding="utf-8") as handle:
            splits = json.load(handle)
        return {
            split: [sample_id for sample_id in sample_ids if sample_id in manifest_sample_ids]
            for split, sample_ids in splits.items()
        }

    with open(cluster_pkl_path, "rb") as handle:
        cluster_data = pickle.load(handle)
    known_clusters = _cluster_source_ids(cluster_data)
    sample_to_cluster = {meta["sample_id"]: str(meta["cluster_id_30"]) for meta in samples_manifest["metadata"]}
    sample_ids = list(sample_to_cluster)
    clusters = sorted({sample_to_cluster[sid] for sid in sample_ids})
    if known_clusters:
        filtered = [cluster for cluster in clusters if cluster in known_clusters]
        if filtered:
            clusters = filtered
    rng = random.Random(seed)
    rng.shuffle(clusters)
    total = len(clusters)
    train_end = int(total * 0.8)
    valid_end = train_end + int(total * 0.1)
    split_clusters = {
        "train": set(clusters[:train_end]),
        "valid": set(clusters[train_end:valid_end]),
        "test": set(clusters[valid_end:]),
    }
    splits = {
        split: [sid for sid in sample_ids if sample_to_cluster[sid] in cluster_ids]
        for split, cluster_ids in split_clusters.items()
    }
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with splits_path.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)
    return splits


def esm_chain_key(pdb_id: str, chain_id: str) -> str:
    return f"{str(pdb_id).lower()}_{str(chain_id)}"


class MuSRNetDataset(Dataset):
    def __init__(
        self,
        manifest: dict[str, Any],
        sample_ids: list[str],
        knn_k: int,
        edge_feature_version: str = "v1",
    ) -> None:
        self.manifest = manifest
        self.sample_ids = sample_ids
        self.knn_k = knn_k
        self.edge_feature_version = edge_feature_version
        self.sample_path_by_id = {sample_id: get_sample_path(manifest, sample_id) for sample_id in manifest["sample_ids"]}

        lmdb_path = manifest.get("esm_lmdb_path")
        if lmdb_path is None:
            raise ValueError("Manifest is missing 'esm_lmdb_path'")
        self.esm_env = open_esm_lmdb(lmdb_path, readonly=True)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        sample = torch.load(self.sample_path_by_id[sample_id], map_location="cpu")
        sample["x_basic"] = build_basic_features(sample)

        wt_key = esm_chain_key(sample["wt_pdb_id"], sample["wt_chain_id"])
        mut_key = esm_chain_key(sample["mut_pdb_id"], sample["mut_chain_id"])
        wt_esm_data = load_chain_esm(self.esm_env, wt_key)
        mut_esm_data = load_chain_esm(self.esm_env, mut_key)

        if wt_esm_data["sequence"] != sample["wt_sequence"]:
            raise ValueError(f"WT ESM sequence mismatch for sample {sample_id}")
        if mut_esm_data["sequence"] != sample["mut_sequence"]:
            raise ValueError(f"Mutant ESM sequence mismatch for sample {sample_id}")

        wt_embedding = wt_esm_data["embedding"].float()
        mut_embedding = mut_esm_data["embedding"].float()
        esm_data = {
            "esm_wt": wt_embedding,
            "esm_delta": mut_embedding - wt_embedding,
        }
        return build_graph(sample, esm_data, self.knn_k, edge_feature_version=self.edge_feature_version)
