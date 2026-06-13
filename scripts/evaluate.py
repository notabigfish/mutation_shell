from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.dataset import MuSRNetDataset, create_or_load_splits, load_samples_manifest
from musrnet.metrics import compute_metrics
from musrnet.models import build_model
from musrnet.train_utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MuSRNet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


@torch.no_grad()
def evaluate_split(model, loader, device):
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
    prediction_rows: list[dict] = []
    for batch in loader:
        batch = batch.to(device)
        outputs = model(
            x_basic=batch.x_basic,
            esm_wt=batch.esm_wt,
            esm_delta=batch.esm_delta,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            shell_id=batch.shell_id,
            batch=batch.batch,
        )
        probs = torch.sigmoid(outputs["perturbed_logit"]).detach().cpu()
        pred_class = outputs["class_logit"].argmax(dim=-1).detach().cpu()
        ptr = batch.ptr.detach().cpu().tolist()
        sample_ids = list(batch.sample_id)
        cluster_ids = list(batch.cluster_id_30)
        disp = outputs["disp"].detach().cpu()
        radius = outputs["radius"].detach().cpu()
        for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
            for residue_index in range(start, end):
                local_index = residue_index - start
                prediction_rows.append(
                    {
                        "sample_id": sample_ids[graph_idx],
                        "cluster_id_30": cluster_ids[graph_idx],
                        "residue_index": local_index,
                        "shell_id": int(batch.shell_id[residue_index].item()),
                        "true_displacement": float(batch.y_disp[residue_index].item()),
                        "pred_displacement": float(disp[residue_index].item()),
                        "true_perturbed": float(batch.y_perturbed[residue_index].item()),
                        "pred_perturbed_prob": float(probs[residue_index].item()),
                        "true_radius": float(batch.y_radius[graph_idx].item()),
                        "pred_radius": float(radius[graph_idx].item()),
                        "true_class": int(batch.y_class[graph_idx].item()),
                        "pred_class": int(pred_class[graph_idx].item()),
                    }
                )
            node_slice = slice(start, end)
            nodes = end - start
            records["true_disp"].extend(batch.y_disp[node_slice].detach().cpu().tolist())
            records["pred_disp"].extend(disp[node_slice].tolist())
            records["shell_id"].extend(batch.shell_id[node_slice].detach().cpu().tolist())
            records["true_perturbed"].extend(batch.y_perturbed[node_slice].detach().cpu().tolist())
            records["pred_perturbed_prob"].extend(probs[node_slice].tolist())
            records["true_radius"].extend([float(batch.y_radius[graph_idx].item())] * nodes)
            records["pred_radius"].extend([float(radius[graph_idx].item())] * nodes)
            records["true_class"].extend([int(batch.y_class[graph_idx].item())] * nodes)
            records["pred_class"].extend([int(pred_class[graph_idx].item())] * nodes)
            records["cluster_id_30"].extend([cluster_ids[graph_idx]] * nodes)
    return compute_metrics(records), prediction_rows


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = config["model_name"]
    model = build_model(model_name, config["model"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    manifest = load_samples_manifest(config["paths"]["samples"])
    cluster_pkl_path = PROJECT_ROOT / "data" / "SingleMutPairs2024_cluster30.pkl"
    splits = create_or_load_splits(manifest, config["paths"]["splits"], cluster_pkl_path, config["seed"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: list[dict] = []
    for split in ["train", "valid", "test"]:
        dataset = MuSRNetDataset(manifest, splits[split], config["data"]["knn_k"])
        loader = DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
        )
        metrics, prediction_rows = evaluate_split(model, loader, device)
        with open(output_dir / f"eval_{split}.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        if split == "test":
            all_predictions = prediction_rows

    csv_path = output_dir / "predictions_test.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "cluster_id_30",
                "residue_index",
                "shell_id",
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
        writer.writerows(all_predictions)


if __name__ == "__main__":
    main()
