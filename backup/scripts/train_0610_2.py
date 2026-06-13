from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any
import random

import torch
import wandb
from tqdm import tqdm
from torch_geometric.data import Batch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, TrainerCallback, ProgressCallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.dataset import MuSRNetDataset, create_or_load_splits, load_samples_manifest
from musrnet.losses import compute_losses
from musrnet.metrics import compute_metrics
from musrnet.model import MuSRNet
from musrnet.seed import set_seed
from musrnet.train_utils import load_yaml, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MuSRNet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--num-workers", type=int, default=-1, help="Number of workers for data loading (overrides config)")
    return parser.parse_args()

def pyg_collate_fn(features: list[Batch]) -> Batch:
    return Batch.from_data_list(features)

def cuda_mem(prefix=""):
    if torch.cuda.is_available():
        print(
            prefix,
            "allocated_GB=",
            round(torch.cuda.memory_allocated() / 1024**3, 3),
            "reserved_GB=",
            round(torch.cuda.memory_reserved() / 1024**3, 3),
            "max_allocated_GB=",
            round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            flush=True,
        )


class MuSRNetTrainer(Trainer):
    def __init__(self, *args, loss_cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_cfg = loss_cfg
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        # DEBUG
        num_graphs = inputs.num_graphs
        num_nodes = inputs.x_basic.shape[0]
        num_edges = inputs.edge_index.shape[1]

        print(
            f"batch graphs={num_graphs}, nodes={num_nodes}, edges={num_edges}, "
            f"esm_wt={tuple(inputs.esm_wt.shape)}, esm_delta={tuple(inputs.esm_delta.shape)}",
            flush=True,
        )
        cuda_mem("before_forward")
        # END DEBUG

        outputs = model(
            x_basic=inputs.x_basic,
            esm_wt=inputs.esm_wt,
            esm_delta=inputs.esm_delta,
            edge_index=inputs.edge_index,
            edge_attr=inputs.edge_attr,
            shell_id=inputs.shell_id,
            batch=inputs.batch,
        )

        cuda_mem("after_forward")  # DEBUG

        loss_dict  = compute_losses(outputs, inputs, **self.loss_cfg)
        self.log(
            {
                "train_disp_loss": loss_dict["disp_loss"].detach().item(),
                "train_pert_loss": loss_dict["pert_loss"].detach().item(),
                "train_radius_loss": loss_dict["radius_loss"].detach().item(),
                "train_class_loss": loss_dict["class_loss"].detach().item(),
            }
        )
        if return_outputs:
            return loss_dict["loss"], outputs
        return loss_dict["loss"]
    
    def get_train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=pyg_collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=pyg_collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

@torch.no_grad()
def evaluate_dataset(model: MuSRNet, dataset: MuSRNetDataset, batch_size: int, num_workers: int, loss_cfg: dict[str, Any]) -> dict[str, float]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pyg_collate_fn,
        num_workers=num_workers,
    )
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
    device = next(model.parameters()).device
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


class EvalCallback(TrainerCallback):
    def __init__(self, valid_dataset, batch_size, num_workers, loss_cfg):
        self.valid_dataset = valid_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loss_cfg = loss_cfg
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        metrics = evaluate_dataset(model, self.valid_dataset, self.batch_size, self.num_workers, self.loss_cfg)
        kwargs['trainer'].log(metrics)
        return control


class EpochProgressBarCallback(TrainerCallback):
    def __init__(self):
        self.training_bar = None
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            steps_per_epoch = int(state.max_steps / args.num_train_epochs)
            current_epoch = int(state.epoch) + 1

            if self.training_bar is None:
                self.training_bar = tqdm(total=steps_per_epoch, desc=f"Epoch {current_epoch}/{args.num_train_epochs}")
            else:
                self.training_bar.reset(total=steps_per_epoch)
                self.training_bar.set_description(f"Epoch {current_epoch}/{args.num_train_epochs}")
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            self.training_bar.update(1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None and logs:
            self.training_bar.set_postfix(logs)

    def on_train_end(self, args, state, control, **kwargs):
        if self.training_bar is not None:
            self.training_bar.close()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(config["seed"])

    samples_manifest = load_samples_manifest(config["paths"]["samples"])
    cluster_pkl_path = PROJECT_ROOT / "data" / "SingleMutPairs2024_cluster30.pkl"
    splits = create_or_load_splits(samples_manifest, config["paths"]["splits"], cluster_pkl_path, config["seed"])

    train_dataset = MuSRNetDataset(samples_manifest, splits["train"], config["data"]["knn_k"])
    valid_dataset = MuSRNetDataset(samples_manifest, splits["valid"], config["data"]["knn_k"])

    model = MuSRNet(**config["model"])

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    project_name = args.project_name or config['wandb']['project_name']
    run_name = args.run_name or config["wandb"]["run_name"]
    os.environ['WANDB_PROJECT'] = project_name
    fp16_enabled = bool(config['train'].get('fp16', True) and torch.cuda.is_available())

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["train"]["epochs"],
        per_device_train_batch_size=config["data"]["batch_size"],
        per_device_eval_batch_size=config["data"]["batch_size"],
        learning_rate=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
        dataloader_num_workers=config["data"]["num_workers"] if args.num_workers < 0 else args.num_workers,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=config["train"].get("logging_steps", 100),
        save_total_limit=config['train'].get('save_total_limit', 5),
        load_best_model_at_end=True,
        metric_for_best_model="eval_shell_mae",
        greater_is_better=False,
        fp16=fp16_enabled,
        report_to=['wandb'],
        run_name=run_name,
        seed=config["seed"],
        remove_unused_columns=False,
        max_grad_norm=config["train"]["grad_clip"],
    )

    save_yaml(output_dir / "config_used.yaml", config)

    trainer = MuSRNetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=pyg_collate_fn,
        loss_cfg=config["loss"],
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config["train"]["patience"]),
            EvalCallback(valid_dataset, config["data"]["batch_size"], config["data"]["num_workers"], config["loss"]),
        ],
    )
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(EpochProgressBarCallback())
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / 'best'))
    trainer.save_state()

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
