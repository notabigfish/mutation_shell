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
import resource



PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.dataset import MuSRNetDataset, create_or_load_splits, load_samples_manifest
from musrnet.losses import compute_losses
from musrnet.metrics import compute_metrics
from musrnet.models import build_model
from musrnet.seed import set_seed
from musrnet.train_utils import load_yaml, save_yaml
from musrnet.labels import derive_class_from_pred

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MuSRNet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--resume-wandb", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--num-workers", type=int, default=-1, help="Number of workers for data loading (overrides config)")
    return parser.parse_args()

def pyg_collate_fn(features: list[Batch]) -> Batch:
    return Batch.from_data_list(features)

def make_pyg_loader(dataset, *, batch_size=None, batch_sampler=None, shuffle=False, num_workers=0, pin_memory=False, prefetch_factor=None, persistent_workers=False):
    kwargs = dict(
        dataset=dataset,
        collate_fn=pyg_collate_fn,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers) and int(num_workers) > 0,
    )
    if batch_sampler is not None:
        kwargs['batch_sampler'] = batch_sampler
    else:
        kwargs['batch_size'] = batch_size
        kwargs['shuffle'] = shuffle
    
    if int(num_workers) > 0:
        kwargs['prefetch_factor'] = int(prefetch_factor or 1)
    return torch.utils.data.DataLoader(**kwargs)
    

class LengthBucketBatchSampler(torch.utils.data.Sampler[list[int]]):
    """
        1. Sort samples by sequence length
        2. Split sorted samples into local buckets
        3. Shuffle buckets and samples inside buckets
        4. Greedily form batches whose total residues <= max_nodes_per_batch
    """
    def __init__(
        self,
        dataset,
        max_nodes_per_batch: int,
        bucket_size: int = 256,
        shuffle: bool = True,
        seed: int = 42,
        max_graphs_per_batch: int | None = None,
        drop_last: bool = False,
    ):
        if max_nodes_per_batch <= 0:
            raise ValueError("max_nodes_per_batch must be positive")
        if bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        
        self.dataset = dataset
        self.max_nodes_per_batch = int(max_nodes_per_batch)
        self.bucket_size = int(bucket_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_graphs_per_batch = max_graphs_per_batch
        self.drop_last = bool(drop_last)
        self.epoch = 0

        length_by_sample_id = {
            str(meta["sample_id"]): int(meta["length"])
            for meta in dataset.manifest["metadata"]
        }

        self.lengths = []
        for sample_id in dataset.sample_ids:
            if sample_id not in length_by_sample_id:
                raise KeyError(f"Missing length metadata for sample_id={sample_id}")
            self.lengths.append(length_by_sample_id[sample_id])

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda idx: self.lengths[idx])

        buckets = [
            indices[i:i + self.bucket_size]
            for i in range(0, len(indices), self.bucket_size)
        ]

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            for bucket in buckets:
                rng.shuffle(bucket)
            rng.shuffle(buckets)
            self.epoch += 1

        ordered_indices = [idx for bucket in buckets for idx in bucket]

        batch = []
        batch_nodes = 0

        for idx in ordered_indices:
            length = self.lengths[idx]

            exceeds_nodes = batch and (batch_nodes + length > self.max_nodes_per_batch)
            exceeds_graphs = (
                self.max_graphs_per_batch is not None
                and batch
                and len(batch) >= self.max_graphs_per_batch
            )

            if exceeds_nodes or exceeds_graphs:
                yield batch
                batch = []
                batch_nodes = 0

            # If a single protein is longer than max_nodes_per_batch,
            # keep it as a one-sample batch instead of dropping it.
            batch.append(idx)
            batch_nodes += length

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        sorted_lengths = sorted(self.lengths)

        num_batches = 0
        batch_nodes = 0
        batch_graphs = 0

        for length in sorted_lengths:
            exceeds_nodes = batch_graphs > 0 and (batch_nodes + length > self.max_nodes_per_batch)
            exceeds_graphs = (
                self.max_graphs_per_batch is not None
                and batch_graphs > 0
                and batch_graphs >= self.max_graphs_per_batch
            )

            if exceeds_nodes or exceeds_graphs:
                num_batches += 1
                batch_nodes = 0
                batch_graphs = 0

            batch_nodes += length
            batch_graphs += 1

        if batch_graphs > 0 and not self.drop_last:
            num_batches += 1

        return num_batches

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
    def __init__(self, *args, loss_cfg, batch_cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_cfg = loss_cfg
        self.batch_cfg = batch_cfg

    def _make_length_batch_sampler(self, dataset, train: bool):
        max_nodes = self.batch_cfg.get("max_nodes_per_batch")

        if max_nodes is None:
            return None

        bucket_size = int(self.batch_cfg.get("length_bucket_size", 256))

        if train:
            max_nodes_per_batch = int(max_nodes)
            shuffle = True
        else:
            max_nodes_per_batch = int(self.batch_cfg.get("eval_max_nodes_per_batch", max_nodes))
            shuffle = False

        # per_device_*_batch_size becomes a safety cap on number of graphs,
        # while max_nodes_per_batch controls the real memory budget.
        max_graphs_per_batch = (
            self.args.per_device_train_batch_size
            if train
            else self.args.per_device_eval_batch_size
        )

        return LengthBucketBatchSampler(
            dataset=dataset,
            max_nodes_per_batch=max_nodes_per_batch,
            bucket_size=bucket_size,
            shuffle=shuffle,
            seed=int(self.args.seed),
            max_graphs_per_batch=max_graphs_per_batch,
            drop_last=False,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        # # DEBUG
        # num_graphs = inputs.num_graphs
        # num_nodes = inputs.x_basic.shape[0]
        # num_edges = inputs.edge_index.shape[1]

        # print(
        #     f"batch graphs={num_graphs}, nodes={num_nodes}, edges={num_edges}, "
        #     f"esm_wt={tuple(inputs.esm_wt.shape)}, esm_delta={tuple(inputs.esm_delta.shape)}",
        #     flush=True,
        # )
        # cuda_mem("before_forward")
        # # END DEBUG

        outputs = model(
            x_basic=inputs.x_basic,
            esm_wt=inputs.esm_wt,
            esm_delta=inputs.esm_delta,
            edge_index=inputs.edge_index,
            edge_attr=inputs.edge_attr,
            shell_id=inputs.shell_id,
            batch=inputs.batch,
        )

        # cuda_mem("after_forward")  # DEBUG
        
        current_epoch = float(self.state.epoch or 0.0)
        loss_dict  = compute_losses(outputs, inputs, current_epoch=current_epoch, **self.loss_cfg)
        
        self.log(
            {
                "train_disp_loss": loss_dict["disp_loss"].detach().item(),
                "train_pert_loss": loss_dict["pert_loss"].detach().item(),
                "train_radius_loss": loss_dict["radius_loss"].detach().item(),
                "train_class_loss": loss_dict["class_loss"].detach().item(),
                "train_w_perturbed_eff": loss_dict['w_perturbed_eff'].detach().item(),
                "train_w_radius_eff": loss_dict["w_radius_eff"].detach().item(),
                "train_w_class_eff": loss_dict["w_class_eff"].detach().item(),                
            }
        )
        if return_outputs:
            return loss_dict["loss"], outputs
        return loss_dict["loss"]
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        device = next(model.parameters()).device
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(
                x_basic=inputs.x_basic,
                esm_wt=inputs.esm_wt,
                esm_delta=inputs.esm_delta,
                edge_index=inputs.edge_index,
                edge_attr=inputs.edge_attr,
                shell_id=inputs.shell_id,
                batch=inputs.batch,
            )
            current_epoch = float(self.state.epoch or 0.0)
            loss_dict  = compute_losses(outputs, inputs, current_epoch=current_epoch, **self.loss_cfg)            
            loss = loss_dict["loss"].detach()

        if prediction_loss_only:
            return loss, None, None

        logits = (
            outputs["disp"].detach(),
            outputs["perturbed_logit"].detach(),
            outputs["radius"].detach(),
            outputs["class_logit"].detach(),
        )
        labels = (
            inputs.y_disp.detach(),
            inputs.y_perturbed.detach(),
            inputs.y_radius.detach(),
            inputs.y_class.detach(),
            inputs.shell_id.detach(),
            inputs.batch.detach(),
        )
        return loss, logits, labels
    
    def get_train_dataloader(self):
        batch_sampler = self._make_length_batch_sampler(self.train_dataset, train=True)
        num_workers = int(self.args.dataloader_num_workers)
        prefetch_factor = int(self.batch_cfg.get('prefecth_factor', 1))
        persistent_workers = bool(self.batch_cfg.get('persistent_workers', False))
        pin_memory = bool(self.batch_cfg.get('pin_memory', False))

        if batch_sampler is not None:
            return make_pyg_loader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,                
            )

        return make_pyg_loader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        batch_sampler = self._make_length_batch_sampler(dataset, train=False)

        num_workers = int(self.batch_cfg.get("eval_num_workers", 0))
        prefetch_factor = int(self.batch_cfg.get("eval_prefetch_factor", 1))
        pin_memory = bool(self.batch_cfg.get("eval_pin_memory", False))

        if batch_sampler is not None:
            return make_pyg_loader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=False,
            )

        return make_pyg_loader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
        )
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        loss_totals, metrics = evaluate_dataset(
            self.model,
            dataset,
            self.args.per_device_eval_batch_size,
            int(self.batch_cfg.get("eval_num_workers", 0)),
            self.loss_cfg,
            self.batch_cfg,
            current_epoch=float(self.state.epoch or 0.0),
        )
        metrics = {
            **{f"{metric_key_prefix}_{k}": v for k, v in loss_totals.items()},
            **{f"{metric_key_prefix}_{k}": v for k, v in metrics.items()},
        }

        self.log(metrics)
        return metrics

@torch.inference_mode()
def evaluate_dataset(model, dataset, batch_size, num_workers, loss_cfg, batch_cfg=None, current_epoch=0.0):
    batch_cfg = batch_cfg or {}
    max_nodes = batch_cfg.get("eval_max_nodes_per_batch", batch_cfg.get("max_nodes_per_batch"))

    if max_nodes is not None:
        batch_sampler = LengthBucketBatchSampler(
            dataset=dataset,
            max_nodes_per_batch=int(max_nodes),
            bucket_size=int(batch_cfg.get("length_bucket_size", 256)),
            shuffle=False,
            seed=int(batch_cfg.get("seed", 42)),
            max_graphs_per_batch=batch_size,
            drop_last=False,
        )
        loader = make_pyg_loader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=int(num_workers),
            pin_memory=bool(batch_cfg.get("eval_pin_memory", False)),
            prefetch_factor=int(batch_cfg.get("eval_prefetch_factor", 1)),
            persistent_workers=False,
        )
    else:
        loader = make_pyg_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=bool(batch_cfg.get("eval_pin_memory", False)),
            prefetch_factor=int(batch_cfg.get("eval_prefetch_factor", 1)),
            persistent_workers=False,
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
        loss_dict = compute_losses(
            outputs,
            batch,
            current_epoch=current_epoch,
            **loss_cfg,
        )

        for key in loss_totals:
            loss_totals[key] += float(loss_dict[key].detach().item())

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
    def __init__(self, valid_dataset, batch_size, num_workers, loss_cfg, batch_cfg):
        self.valid_dataset = valid_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loss_cfg = loss_cfg
        self.batch_cfg = batch_cfg
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        loss_totals, metrics = evaluate_dataset(
            model,
            self.valid_dataset,
            self.batch_size,
            self.num_workers,
            self.loss_cfg,
            self.batch_cfg,
        )

        trainer = kwargs.get("trainer")
        if trainer is not None:
            trainer.log({**loss_totals, **metrics})

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

    model_name = args.model_name or config["model_name"]
    model = build_model(model_name, config["model"])

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    project_name = args.project_name or config['wandb']['project_name']
    run_name = args.run_name or config["wandb"]["run_name"]
    wandb_run_id = args.wandb_run_id or config["wandb"].get("run_id")
    resume_wandb = bool(args.resume_wandb or config["wandb"].get("resume_same_run", False))

    os.environ['WANDB_PROJECT'] = project_name
    
    wandb.init(
        project=project_name,
        name=run_name,
        id=wandb_run_id,
        resume="must" if resume_wandb else False,
        config=config,
    )

    fp16_enabled = bool(config['train'].get('fp16', True) and torch.cuda.is_available())

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["train"]["epochs"],
        per_device_train_batch_size=config["data"]["batch_size"],
        per_device_eval_batch_size=config["data"]["batch_size"],
        learning_rate=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
        dataloader_num_workers=config["data"]["num_workers"] if args.num_workers < 0 else args.num_workers,
        dataloader_pin_memory=bool(config["data"].get("pin_memory", False)),        
        eval_strategy="epoch",
        save_strategy="epoch",
        # eval_strategy='steps',
        # eval_steps=10,
        # save_strategy='steps',
        # save_steps=1000,
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
        batch_cfg=config["data"],
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config["train"]["patience"])
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
