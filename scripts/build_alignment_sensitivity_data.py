from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.dataset import get_sample_path, load_samples_manifest
from musrnet.labels import build_structural_labels
from musrnet.train_utils import load_yaml, save_json, save_yaml

VARIANTS = ["kabsch_exclude_4A", "kabsch_exclude_8A", "kabsch_all", "tmalign"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build alignment-sensitivity label sets for MuSRNet")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--out-config", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tmalign-bin", default=None)
    return parser.parse_args()


def resolve_variants(args: argparse.Namespace) -> list[str]:
    if args.all:
        if args.tmalign_bin:
            return VARIANTS
        print("WARNING: TMALIGN_BIN was not provided; building only Kabsch variants.")
        return VARIANTS[:-1]
    if not args.variant:
        raise ValueError("Either --variant or --all is required")
    return [args.variant]


def update_sample_labels(sample: dict, variant: str, tmalign_bin: str | None) -> dict:
    labels = build_structural_labels(
        coords_wt=sample["coords_wt"].cpu().numpy(),
        coords_mut=sample["coords_mut"].cpu().numpy(),
        mut_pos=int(sample["mut_pos"]),
        alignment_variant=variant,
        tmalign_bin=tmalign_bin,
        sample_id=str(sample["sample_id"]),
    )
    sample["coords_wt_aligned"] = torch.from_numpy(labels["coords_wt_aligned"]).float()
    sample["displacement"] = torch.from_numpy(labels["displacement"]).float()
    sample["radii"] = torch.from_numpy(labels["radii"]).float()
    sample["shell_id"] = torch.from_numpy(labels["shell_id"]).long()
    sample["perturbed"] = torch.from_numpy(labels["perturbed"]).float()
    sample["radius_label"] = torch.from_numpy(labels["radius_label"]).float()
    sample["class_label"] = torch.from_numpy(labels["class_label"]).long()
    sample["alignment_variant"] = labels["alignment_variant"]
    sample["alignment_rmsd"] = float(labels["alignment_rmsd"][0])
    sample["alignment_n_residues"] = int(labels["alignment_n_residues"][0])
    sample["alignment_mask_fraction"] = float(labels["alignment_mask_fraction"][0])
    sample["alignment_metadata"] = labels["alignment_metadata"]
    return sample


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def build_variant(base_config_path: Path, variant: str, out_config_path: Path | None, tmalign_bin: str | None) -> None:
    base_config = load_yaml(base_config_path)
    base_manifest = load_samples_manifest(PROJECT_ROOT / base_config["paths"]["samples"])

    data_dir = PROJECT_ROOT / "data" / "alignment_sensitivity" / variant
    sample_dir = data_dir / "samples"
    manifest_path = data_dir / "samples_manifest.json"
    results_dir = PROJECT_ROOT / "results" / "alignment_sensitivity" / variant
    sample_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    sample_ids: list[str] = []
    metadata: list[dict] = []
    failures: list[dict[str, str]] = []
    alignment_rmsd_values: list[float] = []
    global_disp_values: list[float] = []
    shell_disp: dict[int, list[float]] = {k: [] for k in range(5)}
    shell_pert: dict[int, list[float]] = {k: [] for k in range(5)}

    for sample_id in tqdm(base_manifest["sample_ids"], desc=f"build {variant}"):
        sample_path = get_sample_path(base_manifest, sample_id)
        sample = torch.load(sample_path, map_location="cpu")
        try:
            updated = update_sample_labels(sample, variant, tmalign_bin)
        except Exception as exc:
            failures.append(
                {
                    "sample_id": str(sample_id),
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )
            continue

        out_sample_path = sample_dir / f"{sample_id}.pt"
        torch.save(updated, out_sample_path)
        sample_ids.append(str(sample_id))
        metadata.append(
            {
                "sample_id": str(updated["sample_id"]),
                "cluster_id_30": str(updated["cluster_id_30"]),
                "release_date": str(updated["release_date"]),
                "length": int(updated["coords_wt"].shape[0]),
            }
        )
        alignment_rmsd_values.append(float(updated["alignment_rmsd"]))
        disp_np = updated["displacement"].cpu().numpy()
        shell_np = updated["shell_id"].cpu().numpy()
        pert_np = updated["perturbed"].cpu().numpy()
        global_disp_values.append(float(disp_np.mean()))
        for shell_idx in range(5):
            mask = shell_np == shell_idx
            if mask.any():
                shell_disp[shell_idx].extend(disp_np[mask].tolist())
                shell_pert[shell_idx].extend(pert_np[mask].tolist())

    manifest = dict(base_manifest)
    manifest["samples_dir"] = str(sample_dir)
    manifest["sample_ids"] = sample_ids
    manifest["metadata"] = metadata
    manifest["alignment_variant"] = variant
    with manifest_path.open("w", encoding="utf-8") as handle:
        import json

        json.dump(manifest, handle, indent=2)

    label_stats = {
        "variant": variant,
        "n_samples_requested": len(base_manifest["sample_ids"]),
        "n_samples_written": len(sample_ids),
        "n_failed": len(failures),
        "mean_alignment_rmsd": mean_or_nan(alignment_rmsd_values),
        "median_alignment_rmsd": float(np.median(alignment_rmsd_values)) if alignment_rmsd_values else float("nan"),
        "mean_global_displacement": mean_or_nan(global_disp_values),
        "mean_shell_displacement": {str(k): mean_or_nan(v) for k, v in shell_disp.items()},
        "perturbed_fraction_by_shell": {str(k): mean_or_nan(v) for k, v in shell_pert.items()},
    }
    save_json(results_dir / "label_stats.json", label_stats)

    with (results_dir / "failed_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "error_type", "error_message"])
        writer.writeheader()
        writer.writerows(failures)

    new_config = load_yaml(base_config_path)
    new_config["paths"]["samples"] = str(manifest_path.relative_to(PROJECT_ROOT))
    new_config["paths"]["output_dir"] = f"outputs/c1000/base_v5_align_{variant}"
    new_config.setdefault("wandb", {})
    new_config["wandb"]["run_name"] = f"base_v5_align_{variant}"
    new_config.setdefault("data", {})
    new_config["data"]["alignment_variant"] = variant
    new_config.setdefault("eval", {})
    new_config["eval"]["alignment_variant"] = variant
    if out_config_path is None:
        out_config_path = PROJECT_ROOT / "configs" / "c1000" / f"base_v5_align_{variant}.yaml"
    save_yaml(out_config_path, new_config)


def main() -> None:
    args = parse_args()
    base_config_path = PROJECT_ROOT / args.base_config
    for variant in resolve_variants(args):
        out_config = Path(args.out_config) if args.out_config and not args.all else None
        if out_config is not None and not out_config.is_absolute():
            out_config = PROJECT_ROOT / out_config
        build_variant(base_config_path, variant, out_config, args.tmalign_bin)


if __name__ == "__main__":
    main()
