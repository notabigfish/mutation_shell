from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.dataset import MuSRNetDataset, create_or_load_splits, load_samples_manifest
from musrnet.losses import compute_losses
from musrnet.metrics import compute_metrics
from musrnet.model import MuSRNet
from musrnet.seed import set_seed
from musrnet.train_utils import append_log_row, load_yaml, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MuSRNet")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


@torch.no_grad()
def run_eval(model: MuSRNet, loader: DataLoader, device: torch.device, loss_cfg: dict) -> tuple[dict, dict]:
    model.eval()
    all_records = {
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
    loss_totals = {"loss": 0.0, "disp_loss": 0.0, "pert_loss": 0.0, "radius_loss": 0.0, "class_loss": 0.0}
    batches = 0
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
        losses = compute_losses(outputs, batch, **loss_cfg)
        for key in loss_totals:
            loss_totals[key] += float(losses[key].item())
        batches += 1

        probs = torch.sigmoid(outputs["perturbed_logit"]).detach().cpu()
        pred_class = outputs["class_logit"].argmax(dim=-1).detach().cpu()
        ptr = batch.ptr.detach().cpu().tolist()
        sample_ids = list(batch.sample_id)
        cluster_ids = list(batch.cluster_id_30)
        for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
            node_slice = slice(start, end)
            all_records["true_disp"].extend(batch.y_disp[node_slice].detach().cpu().tolist())
            all_records["pred_disp"].extend(outputs["disp"][node_slice].detach().cpu().tolist())
            all_records["shell_id"].extend(batch.shell_id[node_slice].detach().cpu().tolist())
            all_records["true_perturbed"].extend(batch.y_perturbed[node_slice].detach().cpu().tolist())
            all_records["pred_perturbed_prob"].extend(probs[node_slice].tolist())
            nodes = end - start
            all_records["true_radius"].extend([float(batch.y_radius[graph_idx].item())] * nodes)
            all_records["pred_radius"].extend([float(outputs["radius"][graph_idx].item())] * nodes)
            all_records["true_class"].extend([int(batch.y_class[graph_idx].item())] * nodes)
            all_records["pred_class"].extend([int(pred_class[graph_idx].item())] * nodes)
            all_records["cluster_id_30"].extend([cluster_ids[graph_idx]] * nodes)

    metrics = compute_metrics(all_records)
    if batches:
        for key in loss_totals:
            loss_totals[key] /= batches
    return loss_totals, metrics


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(config["seed"])

    samples_manifest = load_samples_manifest(config["paths"]["samples"])
    cluster_pkl_path = PROJECT_ROOT / "data" / "SingleMutPairs2024_cluster30.pkl"
    splits = create_or_load_splits(samples_manifest, config["paths"]["splits"], cluster_pkl_path, config["seed"])

    datasets = {
        split: MuSRNetDataset(samples_manifest, sample_ids, config["data"]["knn_k"])
        for split, sample_ids in splits.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=(split == "train"),
            num_workers=config["data"]["num_workers"],
        )
        for split, dataset in datasets.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MuSRNet(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(output_dir / "config_used.yaml", config)

    best_metric = float("inf")
    best_epoch = -1
    patience = 0
    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        running = {"loss": 0.0, "disp_loss": 0.0, "pert_loss": 0.0, "radius_loss": 0.0, "class_loss": 0.0}
        num_batches = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                x_basic=batch.x_basic,
                esm_wt=batch.esm_wt,
                esm_delta=batch.esm_delta,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                shell_id=batch.shell_id,
                batch=batch.batch,
            )
            losses = compute_losses(outputs, batch, **config["loss"])
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            optimizer.step()
            for key in running:
                running[key] += float(losses[key].item())
            num_batches += 1

        train_log = {f"train_{key}": value / max(num_batches, 1) for key, value in running.items()}
        valid_losses, valid_metrics = run_eval(model, loaders["valid"], device, config["loss"])
        row = {
            "epoch": epoch,
            **train_log,
            **{f"valid_{key}": value for key, value in valid_losses.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
        }
        append_log_row(output_dir / "train_log.csv", row)

        torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch}, output_dir / "last.pt")
        if valid_metrics["shell_mae"] < best_metric:
            best_metric = valid_metrics["shell_mae"]
            best_epoch = epoch
            patience = 0
            torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch}, output_dir / "best.pt")
        else:
            patience += 1
        print(f"Epoch {epoch}: valid_shell_mae={valid_metrics['shell_mae']:.4f} best_epoch={best_epoch}")
        if patience >= config["train"]["patience"]:
            print("Early stopping triggered")
            break


if __name__ == "__main__":
    main()
