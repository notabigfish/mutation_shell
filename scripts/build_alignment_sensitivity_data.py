from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os

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
    parser.add_argument("--subset", default='c1000')
    parser.add_argument("--workers", type=int, default=max(1, (len(os.sched_getaffinity(0)) or 2) - 2), help="Number of parallel worker processes")
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

def init_worker() -> None:
    torch.set_num_threads(1)


def process_sample(
    sample_id: str,
    sample_path: str,
    sample_dir: str,
    variant: str,
    tmalign_bin: str | None,
) -> dict:
    try:
        sample = torch.load(sample_path, map_location="cpu")
        updated = update_sample_labels(sample, variant, tmalign_bin)

        out_sample_path = Path(sample_dir) / f"{sample_id}.pt"
        torch.save(updated, out_sample_path)

        disp_np = updated["displacement"].cpu().numpy()
        shell_np = updated["shell_id"].cpu().numpy()
        pert_np = updated["perturbed"].cpu().numpy()

        shell_stats: dict[int, dict[str, list[float]]] = {}

        for shell_idx in range(5):
            mask = shell_np == shell_idx
            if mask.any():
                shell_stats[shell_idx] = {
                    "displacement": disp_np[mask].tolist(),
                    "perturbed": pert_np[mask].tolist(),
                }
            else:
                shell_stats[shell_idx] = {
                    "displacement": [],
                    "perturbed": [],
                }

        return {
            "success": True,
            "sample_id": str(sample_id),
            "metadata": {
                "sample_id": str(updated["sample_id"]),
                "cluster_id_30": str(updated["cluster_id_30"]),
                "release_date": str(updated["release_date"]),
                "length": int(updated["coords_wt"].shape[0]),
            },
            "alignment_rmsd": float(updated["alignment_rmsd"]),
            "global_displacement": float(disp_np.mean()),
            "shell_stats": shell_stats,
        }

    except Exception as exc:
        return {
            "success": False,
            "sample_id": str(sample_id),
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }

def build_variant(base_config_path: Path, variant: str, out_config_path: Path | None, tmalign_bin: str | None, subset: str, workers: int) -> None:
    base_config = load_yaml(base_config_path)
    base_manifest = load_samples_manifest(PROJECT_ROOT / base_config["paths"]["samples"])

    data_dir = PROJECT_ROOT / "data" / "alignment_sensitivity" / subset / variant
    sample_dir = data_dir / "samples"
    manifest_path = data_dir / "samples_manifest.json"
    results_dir = PROJECT_ROOT / "outputs" / subset / "alignment_sensitivity" / variant
    sample_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    sample_ids: list[str] = []
    metadata: list[dict] = []
    failures: list[dict[str, str]] = []
    alignment_rmsd_values: list[float] = []
    global_disp_values: list[float] = []
    shell_disp: dict[int, list[float]] = {k: [] for k in range(5)}
    shell_pert: dict[int, list[float]] = {k: [] for k in range(5)}

    future_to_sample_id = {}

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
    ) as executor:
        for sample_id in base_manifest["sample_ids"]:
            sample_path = get_sample_path(base_manifest, sample_id)

            future = executor.submit(
                process_sample,
                str(sample_id),
                str(sample_path),
                str(sample_dir),
                variant,
                tmalign_bin,
            )
            future_to_sample_id[future] = str(sample_id)

        with tqdm(
            total=len(future_to_sample_id),
            desc=f"build {variant}",
        ) as progress:
            for future in as_completed(future_to_sample_id):
                sample_id = future_to_sample_id[future]

                try:
                    result = future.result()
                except Exception as exc:
                    failures.append(
                        {
                            "sample_id": sample_id,
                            "error_type": exc.__class__.__name__,
                            "error_message": str(exc),
                        }
                    )
                    progress.update(1)
                    continue

                if not result["success"]:
                    failures.append(
                        {
                            "sample_id": result["sample_id"],
                            "error_type": result["error_type"],
                            "error_message": result["error_message"],
                        }
                    )
                    progress.update(1)
                    continue

                sample_ids.append(result["sample_id"])
                metadata.append(result["metadata"])
                alignment_rmsd_values.append(result["alignment_rmsd"])
                global_disp_values.append(result["global_displacement"])

                for shell_idx in range(5):
                    stats = result["shell_stats"][shell_idx]
                    shell_disp[shell_idx].extend(stats["displacement"])
                    shell_pert[shell_idx].extend(stats["perturbed"])

                progress.update(1)
    metadata_by_id = {item["sample_id"]: item for item in metadata}
    sample_ids = [str(sample_id) for sample_id in base_manifest["sample_ids"] if str(sample_id) in metadata_by_id]
    metadata = [metadata_by_id[sample_id] for sample_id in sample_ids]

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
    new_config["paths"]["output_dir"] = f"outputs/{subset}/base_v5_align_{variant}"
    new_config.setdefault("wandb", {})
    new_config["wandb"]["run_name"] = f"base_v5_align_{variant}"
    new_config.setdefault("data", {})
    new_config["data"]["alignment_variant"] = variant
    new_config.setdefault("eval", {})
    new_config["eval"]["alignment_variant"] = variant
    if out_config_path is None:
        out_config_path = PROJECT_ROOT / "configs" / subset / f"base_v5_align_{variant}.yaml"
    save_yaml(out_config_path, new_config)


def main() -> None:
    args = parse_args()
    base_config_path = PROJECT_ROOT / args.base_config
    for variant in resolve_variants(args):
        out_config = Path(args.out_config) if args.out_config and not args.all else None
        if out_config is not None and not out_config.is_absolute():
            out_config = PROJECT_ROOT / out_config
        build_variant(base_config_path, variant, out_config, args.tmalign_bin, args.subset, args.workers)


if __name__ == "__main__":
    main()
