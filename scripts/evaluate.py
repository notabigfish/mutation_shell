from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.dataset import MuSRNetDataset, create_or_load_splits, load_samples_manifest
from musrnet.eval_utils import derive_radius_and_class_for_graph
from musrnet.metrics import compute_metrics
from musrnet.models import build_model
from musrnet.train_utils import load_yaml
from scripts.train import LengthBucketBatchSampler, make_pyg_loader

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MuSRNet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def forward_model(model, batch):
    batch_inputs = {
        "x_basic": batch.x_basic,
        "esm_wt": batch.esm_wt,
        "esm_delta": batch.esm_delta,
        "edge_index": batch.edge_index,
        "edge_attr": batch.edge_attr,
        "shell_id": batch.shell_id,
        "batch": batch.batch,
    }
    for optional_name in ["radii", "is_mutation_site", "mut_pos", "mutation_key"]:
        if hasattr(batch, optional_name):
            batch_inputs[optional_name] = getattr(batch, optional_name)
    supported = inspect.signature(model.forward).parameters
    return model(**{key: value for key, value in batch_inputs.items() if key in supported})


def load_checkpoint_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        return load_file(str(checkpoint_path), device="cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    return checkpoint


@torch.no_grad()
def evaluate_split(model, loader, device, eval_cfg: dict):
    model.eval()
    records = {
        "true_disp": [],
        "pred_disp": [],
        "shell_id": [],
        "true_perturbed": [],
        "pred_perturbed_prob": [],
        "true_radius": [],
        "pred_radius": [],
        "true_class": [],
        "pred_class": [],
        "cluster_id_30": [],
    }
    prediction_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            for batch in tqdm(loader, desc="Evaluating"):
                batch = batch.to(device)
                outputs = forward_model(model, batch)
                probs = torch.sigmoid(outputs["perturbed_logit"]).detach().cpu()
                disp = outputs["disp"].detach().cpu()
                ptr = batch.ptr.detach().cpu().tolist()
                sample_ids = list(batch.sample_id)
                cluster_ids = list(batch.cluster_id_30)
                radii = batch.radii.detach().cpu()
                for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
                    node_slice = slice(start, end)
                    pred_radius, pred_class = derive_radius_and_class_for_graph(
                        pred_disp=disp[node_slice],
                        pred_perturbed_prob=probs[node_slice],
                        radii=radii[node_slice],
                        displacement_threshold=float(eval_cfg.get("displacement_threshold", 1.0)),
                        response_threshold=float(eval_cfg.get("response_threshold", 0.5)),
                        radius_threshold=float(eval_cfg.get("radius_threshold", 8.0)),
                    )
                    true_radius = float(batch.y_radius[graph_idx].detach().cpu().item())
                    true_class = int(batch.y_class[graph_idx].detach().cpu().item())
                    nodes = end - start
                    records["true_disp"].extend(batch.y_disp[node_slice].detach().cpu().tolist())
                    records["pred_disp"].extend(disp[node_slice].tolist())
                    records["shell_id"].extend(batch.shell_id[node_slice].detach().cpu().tolist())
                    records["true_perturbed"].extend(batch.y_perturbed[node_slice].detach().cpu().tolist())
                    records["pred_perturbed_prob"].extend(probs[node_slice].tolist())
                    records["true_radius"].extend([true_radius] * nodes)
                    records["pred_radius"].extend([pred_radius] * nodes)
                    records["true_class"].extend([true_class] * nodes)
                    records["pred_class"].extend([pred_class] * nodes)
                    records["cluster_id_30"].extend([cluster_ids[graph_idx]] * nodes)
                    for residue_index in range(start, end):
                        local_index = residue_index - start
                        prediction_rows.append(
                            {
                                "sample_id": sample_ids[graph_idx],
                                "cluster_id_30": cluster_ids[graph_idx],
                                "residue_index": local_index,
                                "shell_id": int(batch.shell_id[residue_index].item()),
                                "radii": float(batch.radii[residue_index].item()),
                                "true_displacement": float(batch.y_disp[residue_index].item()),
                                "pred_displacement": float(disp[residue_index].item()),
                                "true_perturbed": float(batch.y_perturbed[residue_index].item()),
                                "pred_perturbed_prob": float(probs[residue_index].item()),
                                "true_radius": true_radius,
                                "pred_radius": pred_radius,
                                "true_class": true_class,
                                "pred_class": pred_class,
                            }
                        )
    return compute_metrics(records), prediction_rows

def write_predictions(csv_path: Path, rows: list[dict[str, object]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "cluster_id_30",
                "residue_index",
                "shell_id",
                "radii",
                "true_displacement",
                "pred_displacement",
                "true_perturbed",
                "pred_perturbed_prob",
                "true_radius",
                "pred_radius",
                "true_class",
                "pred_class",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    model = build_model(config["model_name"], config["model"])
    model.load_state_dict(load_checkpoint_state_dict(args.checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    manifest = load_samples_manifest(config["paths"]["samples"])
    cluster_pkl_path = PROJECT_ROOT / "data" / "SingleMutPairs2024_cluster30.pkl"
    splits = create_or_load_splits(manifest, config["paths"]["splits"], cluster_pkl_path, config["seed"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_feature_version = config["data"].get("edge_feature_version", "v1")

    # for split in ["train", "valid", "test"]:
    for split in ["test"]:
        dataset = MuSRNetDataset(manifest, splits[split], config["data"]["knn_k"], edge_feature_version=edge_feature_version)
        batch_sampler = LengthBucketBatchSampler(
            dataset=dataset,
            max_nodes_per_batch=int(config["data"].get("eval_max_nodes_per_batch", config["data"]["max_nodes_per_batch"])),
            bucket_size=int(config["data"].get("length_bucket_size", 256)),
            shuffle=False,
            seed=int(config.get("seed", 42)),
            max_graphs_per_batch=int(config["data"]["batch_size"]),
            drop_last=False,
        )
        loader = make_pyg_loader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=int(config["data"].get("eval_num_workers", 0)),
        pin_memory=bool(config["data"].get("eval_pin_memory", False)),
        prefetch_factor=int(config["data"].get("eval_prefetch_factor", 1)),
        persistent_workers=False,
        )
        metrics, prediction_rows = evaluate_split(model, loader, device, config.get("eval", {}))
        with (output_dir / f"eval_{split}.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        write_predictions(output_dir / f"predictions_{split}.csv", prediction_rows)


if __name__ == "__main__":
    main()
