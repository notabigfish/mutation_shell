from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.dataset import MuSRNetDataset, load_samples_manifest
from musrnet.models import build_model
from musrnet.seed import set_seed
from musrnet.train_utils import load_yaml


def collate_fn(items):
    return Batch.from_data_list(items)


def shell_balanced_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    shell_id: torch.Tensor,
    batch: torch.Tensor,
    delta: float = 1.0,
    num_shells: int = 5,
) -> torch.Tensor:
    err = pred - target
    abs_err = err.abs()

    huber = torch.where(
        abs_err <= delta,
        0.5 * err.pow(2),
        delta * (abs_err - 0.5 * delta),
    )

    num_graphs = int(batch.max().item()) + 1
    losses = []

    for graph_idx in range(num_graphs):
        graph_mask = batch == graph_idx
        graph_losses = []

        for shell_idx in range(num_shells):
            mask = graph_mask & (shell_id == shell_idx)
            if mask.any():
                graph_losses.append(huber[mask].mean())

        if graph_losses:
            losses.append(torch.stack(graph_losses).mean())

    return torch.stack(losses).mean()


def shell_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    shell_id: torch.Tensor,
    batch: torch.Tensor,
    num_shells: int = 5,
) -> torch.Tensor:
    abs_err = (pred - target).abs()
    num_graphs = int(batch.max().item()) + 1
    losses = []

    for graph_idx in range(num_graphs):
        graph_mask = batch == graph_idx
        graph_losses = []

        for shell_idx in range(num_shells):
            mask = graph_mask & (shell_id == shell_idx)
            if mask.any():
                graph_losses.append(abs_err[mask].mean())

        if graph_losses:
            losses.append(torch.stack(graph_losses).mean())

    return torch.stack(losses).mean()


def evaluate_train_set(model, loader, device):
    model.eval()

    total_disp = 0.0
    total_pert = 0.0
    total_shell_mae = 0.0
    total_global_mae = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(
                x_basic=batch.x_basic,
                esm_wt=batch.esm_wt,
                esm_delta=batch.esm_delta,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                shell_id=batch.shell_id,
                batch=batch.batch,
            )

            disp_loss = shell_balanced_huber_loss(
                out["disp"],
                batch.y_disp,
                batch.shell_id,
                batch.batch,
            )

            pert_loss = F.binary_cross_entropy_with_logits(
                out["perturbed_logit"],
                batch.y_perturbed.float(),
            )

            s_mae = shell_mae(
                out["disp"],
                batch.y_disp,
                batch.shell_id,
                batch.batch,
            )

            g_mae = (out["disp"] - batch.y_disp).abs().mean()

            total_disp += float(disp_loss.item())
            total_pert += float(pert_loss.item())
            total_shell_mae += float(s_mae.item())
            total_global_mae += float(g_mae.item())
            n_batches += 1

    return {
        "disp_loss": total_disp / n_batches,
        "pert_loss": total_pert / n_batches,
        "shell_mae": total_shell_mae / n_batches,
        "global_mae": total_global_mae / n_batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--w-pert", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_yaml(args.config)
    set_seed(int(config["seed"]))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    samples_manifest = load_samples_manifest(config["paths"]["samples"])
    sample_ids = [m["sample_id"] for m in samples_manifest["metadata"][: args.num_samples]]

    dataset = MuSRNetDataset(samples_manifest, sample_ids, int(config["data"]["knn_k"]))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model_cfg = dict(config["model"])
    model_cfg["dropout"] = 0.0

    model_name = config["model_name"]
    model = build_model(model_name, model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )

    init_metrics = evaluate_train_set(model, eval_loader, device)
    print("INIT:", init_metrics)

    model.train()
    step = 0
    pbar = tqdm(total=args.steps)

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            batch = batch.to(device)

            out = model(
                x_basic=batch.x_basic,
                esm_wt=batch.esm_wt,
                esm_delta=batch.esm_delta,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                shell_id=batch.shell_id,
                batch=batch.batch,
            )

            disp_loss = shell_balanced_huber_loss(
                out["disp"],
                batch.y_disp,
                batch.shell_id,
                batch.batch,
            )

            pert_loss = F.binary_cross_entropy_with_logits(
                out["perturbed_logit"],
                batch.y_perturbed.float(),
            )

            loss = disp_loss + args.w_pert * pert_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 100 == 0:
                pbar.set_postfix(
                    {
                        "loss": float(loss.item()),
                        "disp": float(disp_loss.item()),
                        "pert": float(pert_loss.item()),
                    }
                )

            step += 1
            pbar.update(1)

    pbar.close()

    final_metrics = evaluate_train_set(model, eval_loader, device)
    print("FINAL:", final_metrics)

    shell_ratio = final_metrics["shell_mae"] / max(init_metrics["shell_mae"], 1e-8)
    pert_ratio = final_metrics["pert_loss"] / max(init_metrics["pert_loss"], 1e-8)

    print("RATIO:", {"shell_mae_ratio": shell_ratio, "pert_loss_ratio": pert_ratio})

    if shell_ratio > 0.5:
        print("FAILED: shell_mae did not decrease enough. Check labels, loss, edge direction, or model forward.")
        raise SystemExit(1)

    if pert_ratio > 0.7:
        print("FAILED: pert_loss did not decrease enough. Check perturbed labels or perturbed head.")
        raise SystemExit(1)

    print("PASSED: model can overfit 32 samples.")


if __name__ == "__main__":
    main()