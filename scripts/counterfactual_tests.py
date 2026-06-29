from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.constants import AA_TO_INDEX, AMINO_ACIDS, RBF_CENTERS, RBF_SIGMA
from musrnet.dataset import MuSRNetDataset, create_or_load_splits, load_samples_manifest
from musrnet.eval_utils import derive_radius_and_class_for_graph
from musrnet.graph import build_edge_attr_v1, build_edge_attr_v4, knn_edges
from musrnet.metrics import compute_metrics
from musrnet.models import build_model
from musrnet.train_utils import load_yaml
from scripts.train import LengthBucketBatchSampler, make_pyg_loader

AA_START = 0
AA_END = 20
MUT_SITE_COL = 20
MUT_VEC_START = 21
MUT_VEC_END = 41
RADIAL_START = 41
RADIAL_END = 57

VARIANTS = [
    "original",
    "shuffle_site",
    "shuffle_mutant_aa",
    "wt_wt_negative",
    "reverse_mutation",
    "remove_mutation_context",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference-time counterfactual tests for MuSRNet base_v5")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--response-threshold", type=float, default=None)
    parser.add_argument("--displacement-threshold", type=float, default=None)
    parser.add_argument("--radius-threshold", type=float, default=None)
    parser.add_argument("--mut-aa-esm-mode", choices=["reuse_delta", "negate_delta"], default="reuse_delta")
    return parser.parse_args()


def load_checkpoint_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        return load_file(str(checkpoint_path), device="cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    return checkpoint


def aa_one_hot(aa: str) -> torch.Tensor:
    vec = torch.zeros(20, dtype=torch.float32)
    vec[AA_TO_INDEX[aa]] = 1.0
    return vec


def rbf_torch(values: torch.Tensor) -> torch.Tensor:
    centers = torch.tensor(RBF_CENTERS, device=values.device, dtype=values.dtype)
    return torch.exp(-((values[:, None] - centers[None, :]) ** 2) / (2.0 * (RBF_SIGMA**2)))


def compute_shell_ids_from_radii(radii: torch.Tensor) -> torch.Tensor:
    shell = torch.full_like(radii, fill_value=4, dtype=torch.long)
    shell[radii <= 16.0] = 3
    shell[radii <= 12.0] = 2
    shell[radii <= 8.0] = 1
    shell[radii <= 4.0] = 0
    return shell


def stable_int_hash(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def clone_data(data: Data) -> Data:
    cloned = copy.copy(data)
    for key in data.keys():
        value = getattr(data, key)
        if torch.is_tensor(value):
            setattr(cloned, key, value.clone())
        else:
            setattr(cloned, key, copy.deepcopy(value))
    return cloned


def save_eval_fields(data: Data) -> None:
    data.eval_shell_id = data.shell_id.clone()
    data.eval_radii = data.radii.clone()
    data.eval_y_disp = data.y_disp.clone()
    data.eval_y_perturbed = data.y_perturbed.clone()
    data.eval_y_radius = data.y_radius.clone()
    data.eval_y_class = data.y_class.clone()


def rebuild_graph_for_mutation_site(data: Data, new_mut_pos: int, knn_k: int, edge_feature_version: str) -> Data:
    coords_np = data.pos.detach().cpu().numpy()
    edge_index_np, edge_distances = knn_edges(coords_np, knn_k)

    x_mut = data.pos[new_mut_pos]
    radii = torch.linalg.norm(data.pos - x_mut[None, :], dim=-1)
    shell_id = compute_shell_ids_from_radii(radii)

    if edge_feature_version == "v4":
        edge_attr_np = build_edge_attr_v4(
            coords=coords_np,
            edge_index_np=edge_index_np,
            edge_distances=edge_distances,
            mut_pos=int(new_mut_pos),
            shell_id=shell_id.cpu().numpy(),
        )
    else:
        edge_attr_np = build_edge_attr_v1(edge_index_np, edge_distances)

    data.edge_index = torch.from_numpy(edge_index_np).long()
    data.edge_attr = torch.from_numpy(edge_attr_np).float()
    data.radii = radii.float()
    data.shell_id = shell_id.long()
    data.mut_pos = torch.tensor([int(new_mut_pos)], dtype=torch.long)

    data.x_basic[:, MUT_SITE_COL] = 0.0
    data.x_basic[new_mut_pos, MUT_SITE_COL] = 1.0
    data.is_mutation_site = data.x_basic[:, MUT_SITE_COL].float()
    data.x_basic[:, RADIAL_START:RADIAL_END] = rbf_torch(radii.float()).cpu()
    return data


def apply_original(data: Data) -> Data:
    save_eval_fields(data)
    return data


def apply_shuffle_site(data: Data, *, seed: int, knn_k: int, edge_feature_version: str) -> Data:
    save_eval_fields(data)
    n = data.x_basic.size(0)
    old_pos = int(data.mut_pos.view(-1)[0].item())
    rng = random.Random(seed + stable_int_hash(str(data.sample_id)))
    candidates = [i for i in range(n) if i != old_pos]
    new_pos = rng.choice(candidates)
    data = rebuild_graph_for_mutation_site(data, new_mut_pos=new_pos, knn_k=knn_k, edge_feature_version=edge_feature_version)
    data.counterfactual_old_mut_pos = old_pos
    data.counterfactual_new_mut_pos = new_pos
    return data


def apply_shuffle_mutant_aa(data: Data, *, seed: int, mut_aa_esm_mode: str) -> Data:
    save_eval_fields(data)
    wt_aa = str(data.wt_aa)
    true_mut_aa = str(data.mut_aa)
    rng = random.Random(seed + stable_int_hash(str(data.sample_id)))
    candidates = [aa for aa in AMINO_ACIDS if aa != wt_aa and aa != true_mut_aa]
    fake_mut_aa = rng.choice(candidates)

    wt_vec = aa_one_hot(wt_aa)
    fake_mut_vec = aa_one_hot(fake_mut_aa)
    fake_mutation_vector = fake_mut_vec - wt_vec
    data.x_basic[:, MUT_VEC_START:MUT_VEC_END] = fake_mutation_vector[None, :]

    if mut_aa_esm_mode == "negate_delta":
        data.esm_delta = -data.esm_delta
    elif mut_aa_esm_mode == "reuse_delta":
        pass
    else:
        raise ValueError(mut_aa_esm_mode)

    data.counterfactual_true_mut_aa = true_mut_aa
    data.counterfactual_fake_mut_aa = fake_mut_aa
    return data


def apply_wt_wt_negative(data: Data) -> Data:
    save_eval_fields(data)
    data.x_basic[:, MUT_VEC_START:MUT_VEC_END] = 0.0
    data.esm_delta = torch.zeros_like(data.esm_delta)
    data.eval_y_disp = torch.zeros_like(data.y_disp)
    data.eval_y_perturbed = torch.zeros_like(data.y_perturbed)
    data.eval_y_radius = torch.zeros_like(data.y_radius)
    data.eval_y_class = torch.zeros_like(data.y_class)
    return data


def apply_reverse_mutation(data: Data) -> Data:
    save_eval_fields(data)
    wt_aa = str(data.wt_aa)
    mut_aa = str(data.mut_aa)
    wt_vec = aa_one_hot(wt_aa)
    mut_vec = aa_one_hot(mut_aa)
    reverse_vector = wt_vec - mut_vec
    data.x_basic[:, MUT_VEC_START:MUT_VEC_END] = reverse_vector[None, :]
    data.esm_delta = -data.esm_delta
    data.counterfactual_reverse_key = f"{mut_aa}->{wt_aa}"
    return data


def apply_remove_mutation_context(data: Data) -> Data:
    save_eval_fields(data)
    return data


class CounterfactualDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        variant: str,
        seed: int,
        knn_k: int,
        edge_feature_version: str,
        mut_aa_esm_mode: str,
    ) -> None:
        self.base_dataset = base_dataset
        self.variant = variant
        self.seed = seed
        self.knn_k = knn_k
        self.edge_feature_version = edge_feature_version
        self.mut_aa_esm_mode = mut_aa_esm_mode

        self.manifest = base_dataset.manifest
        self.sample_ids = base_dataset.sample_ids

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        data = clone_data(self.base_dataset[index])
        assert data.x_basic.size(1) >= 57
        if self.variant == "original":
            return apply_original(data)
        if self.variant == "shuffle_site":
            return apply_shuffle_site(data, seed=self.seed, knn_k=self.knn_k, edge_feature_version=self.edge_feature_version)
        if self.variant == "shuffle_mutant_aa":
            return apply_shuffle_mutant_aa(data, seed=self.seed, mut_aa_esm_mode=self.mut_aa_esm_mode)
        if self.variant == "wt_wt_negative":
            return apply_wt_wt_negative(data)
        if self.variant == "reverse_mutation":
            return apply_reverse_mutation(data)
        if self.variant == "remove_mutation_context":
            return apply_remove_mutation_context(data)
        raise ValueError(f"Unknown counterfactual variant: {self.variant}")


def forward_model(model, batch, *, disable_mutation_context: bool = False):
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
    if "disable_mutation_context" in supported:
        batch_inputs["disable_mutation_context"] = disable_mutation_context
    return model(**{k: v for k, v in batch_inputs.items() if k in supported})


@torch.no_grad()
def evaluate_counterfactual_variant(model, loader, device, eval_cfg, variant: str):
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
    graph_pred_classes: list[int] = []
    disable_mutation_context = variant == "remove_mutation_context"

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            for batch in tqdm(loader, desc=f"Evaluating {variant}"):
                batch = batch.to(device)
                outputs = forward_model(model, batch, disable_mutation_context=disable_mutation_context)
                probs = torch.sigmoid(outputs["perturbed_logit"]).detach().cpu()
                disp = outputs["disp"].detach().cpu()

                metric_shell_id = batch.eval_shell_id.detach().cpu() if hasattr(batch, "eval_shell_id") else batch.shell_id.detach().cpu()
                metric_radii = batch.eval_radii.detach().cpu() if hasattr(batch, "eval_radii") else batch.radii.detach().cpu()
                true_disp = batch.eval_y_disp.detach().cpu() if hasattr(batch, "eval_y_disp") else batch.y_disp.detach().cpu()
                true_perturbed = batch.eval_y_perturbed.detach().cpu() if hasattr(batch, "eval_y_perturbed") else batch.y_perturbed.detach().cpu()
                true_radius = batch.eval_y_radius.detach().cpu() if hasattr(batch, "eval_y_radius") else batch.y_radius.detach().cpu()
                true_class = batch.eval_y_class.detach().cpu() if hasattr(batch, "eval_y_class") else batch.y_class.detach().cpu()

                ptr = batch.ptr.detach().cpu().tolist()
                sample_ids = list(batch.sample_id)
                cluster_ids = list(batch.cluster_id_30)

                for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
                    node_slice = slice(start, end)
                    pred_radius, pred_class = derive_radius_and_class_for_graph(
                        pred_disp=disp[node_slice],
                        pred_perturbed_prob=probs[node_slice],
                        radii=metric_radii[node_slice],
                        displacement_threshold=float(eval_cfg.get("displacement_threshold", 1.0)),
                        response_threshold=float(eval_cfg.get("response_threshold", 0.5)),
                        radius_threshold=float(eval_cfg.get("radius_threshold", 8.0)),
                    )
                    graph_pred_classes.append(pred_class)

                    graph_true_radius = float(true_radius[graph_idx].item())
                    graph_true_class = int(true_class[graph_idx].item())
                    nodes = end - start
                    records["true_disp"].extend(true_disp[node_slice].tolist())
                    records["pred_disp"].extend(disp[node_slice].tolist())
                    records["shell_id"].extend(metric_shell_id[node_slice].tolist())
                    records["true_perturbed"].extend(true_perturbed[node_slice].tolist())
                    records["pred_perturbed_prob"].extend(probs[node_slice].tolist())
                    records["true_radius"].extend([graph_true_radius] * nodes)
                    records["pred_radius"].extend([pred_radius] * nodes)
                    records["true_class"].extend([graph_true_class] * nodes)
                    records["pred_class"].extend([pred_class] * nodes)
                    records["cluster_id_30"].extend([cluster_ids[graph_idx]] * nodes)

                    for residue_index in range(start, end):
                        local_index = residue_index - start
                        prediction_rows.append(
                            {
                                "variant": variant,
                                "sample_id": sample_ids[graph_idx],
                                "cluster_id_30": cluster_ids[graph_idx],
                                "residue_index": local_index,
                                "shell_id": int(metric_shell_id[residue_index].item()),
                                "radii": float(metric_radii[residue_index].item()),
                                "true_displacement": float(true_disp[residue_index].item()),
                                "pred_displacement": float(disp[residue_index].item()),
                                "true_perturbed": float(true_perturbed[residue_index].item()),
                                "pred_perturbed_prob": float(probs[residue_index].item()),
                                "true_radius": graph_true_radius,
                                "pred_radius": pred_radius,
                                "true_class": graph_true_class,
                                "pred_class": pred_class,
                            }
                        )

    metrics = compute_metrics(records)
    pred_disp_np = np.asarray(records["pred_disp"], dtype=np.float32)
    pred_prob_np = np.asarray(records["pred_perturbed_prob"], dtype=np.float32)
    pred_score_np = pred_disp_np * pred_prob_np
    response_threshold = float(eval_cfg.get("response_threshold", 0.5))
    pred_perturbed_mask = pred_score_np > response_threshold

    metrics["mean_pred_displacement"] = float(np.mean(pred_disp_np)) if pred_disp_np.size else float("nan")
    metrics["max_pred_displacement"] = float(np.max(pred_disp_np)) if pred_disp_np.size else float("nan")
    metrics["mean_pred_perturbed_prob"] = float(np.mean(pred_prob_np)) if pred_prob_np.size else float("nan")
    metrics["pred_perturbed_rate"] = float(np.mean(pred_perturbed_mask.astype(np.float32))) if pred_perturbed_mask.size else float("nan")
    metrics["pred_perturbed_rate_at_score_threshold"] = metrics["pred_perturbed_rate"]

    if graph_pred_classes:
        graph_pred_classes_np = np.asarray(graph_pred_classes, dtype=np.int64)
        for class_idx in range(3):
            metrics[f"pred_class_{class_idx}_rate"] = float(np.mean(graph_pred_classes_np == class_idx))
        metrics["silent_prediction_rate"] = metrics["pred_class_0_rate"]
    else:
        for class_idx in range(3):
            metrics[f"pred_class_{class_idx}_rate"] = float("nan")
        metrics["silent_prediction_rate"] = float("nan")

    return metrics, prediction_rows


def write_predictions(csv_path: Path, rows: list[dict[str, object]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
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


def write_summary_csv(csv_path: Path, summary_rows: list[dict[str, object]]) -> None:
    fieldnames = []
    for row in summary_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def build_summary_rows(variant_metrics: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    original = variant_metrics["original"]
    rows: list[dict[str, object]] = []
    for variant, metrics in variant_metrics.items():
        row: dict[str, object] = {
            "variant": variant,
            "global_mae": metrics.get("global_mae"),
            "shell_mae": metrics.get("shell_mae"),
            "cluster_avg_shell_mae": metrics.get("cluster_avg_shell_mae"),
            "mae_shell_0": metrics.get("mae_shell_0"),
            "mae_shell_1": metrics.get("mae_shell_1"),
            "mae_shell_2": metrics.get("mae_shell_2"),
            "mae_shell_3": metrics.get("mae_shell_3"),
            "mae_shell_4": metrics.get("mae_shell_4"),
            "perturbed_auroc": metrics.get("perturbed_auroc"),
            "perturbed_auprc": metrics.get("perturbed_auprc"),
            "radius_mae": metrics.get("radius_mae"),
            "class_macro_f1": metrics.get("class_macro_f1"),
            "mean_pred_displacement": metrics.get("mean_pred_displacement"),
            "max_pred_displacement": metrics.get("max_pred_displacement"),
            "mean_pred_perturbed_prob": metrics.get("mean_pred_perturbed_prob"),
            "pred_perturbed_rate": metrics.get("pred_perturbed_rate"),
            "pred_class_0_rate": metrics.get("pred_class_0_rate"),
            "pred_class_1_rate": metrics.get("pred_class_1_rate"),
            "pred_class_2_rate": metrics.get("pred_class_2_rate"),
        }
        row["delta_shell_mae"] = metrics.get("shell_mae", float("nan")) - original.get("shell_mae", float("nan"))
        row["delta_global_mae"] = metrics.get("global_mae", float("nan")) - original.get("global_mae", float("nan"))
        row["delta_auprc"] = metrics.get("perturbed_auprc", float("nan")) - original.get("perturbed_auprc", float("nan"))
        row["delta_auroc"] = metrics.get("perturbed_auroc", float("nan")) - original.get("perturbed_auroc", float("nan"))
        row["degraded_shell_mae"] = bool(row["delta_shell_mae"] > 0) if variant != "original" else False
        row["degraded_global_mae"] = bool(row["delta_global_mae"] > 0) if variant != "original" else False
        row["degraded_auprc"] = bool(row["delta_auprc"] < 0) if variant != "original" else False
        row["degraded_auroc"] = bool(row["delta_auroc"] < 0) if variant != "original" else False
        if variant == "wt_wt_negative":
            row["wt_wt_low_mean_disp"] = bool(metrics.get("mean_pred_displacement", float("inf")) < 0.2)
            row["wt_wt_low_perturbed_rate"] = bool(metrics.get("pred_perturbed_rate", float("inf")) < 0.05)
            row["wt_wt_mostly_silent"] = bool(metrics.get("pred_class_0_rate", 0.0) > 0.8)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    eval_cfg = dict(config.get("eval", {}))
    if args.response_threshold is not None:
        eval_cfg["response_threshold"] = args.response_threshold
    if args.displacement_threshold is not None:
        eval_cfg["displacement_threshold"] = args.displacement_threshold
    if args.radius_threshold is not None:
        eval_cfg["radius_threshold"] = args.radius_threshold

    if args.mut_aa_esm_mode == "reuse_delta":
        print(
            "WARNING: shuffle_mutant_aa corrupts explicit mutation vector only unless --mut-aa-esm-mode negate_delta is used. "
            "It does not recompute ESM embeddings."
        )

    model = build_model(config["model_name"], config["model"])
    model.load_state_dict(load_checkpoint_state_dict(args.checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    manifest = load_samples_manifest(config["paths"]["samples"])
    cluster_pkl_path = PROJECT_ROOT / "data" / "SingleMutPairs2024_cluster30.pkl"
    splits = create_or_load_splits(manifest, config["paths"]["splits"], cluster_pkl_path, config["seed"])
    edge_feature_version = config["data"].get("edge_feature_version", "v1")
    dataset = MuSRNetDataset(manifest, splits[args.split], config["data"]["knn_k"], edge_feature_version=edge_feature_version)
    if args.max_samples is not None:
        dataset.sample_ids = dataset.sample_ids[: args.max_samples]

    batch_size = args.batch_size or config["data"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else config["data"]["num_workers"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variant_metrics: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        cf_dataset = CounterfactualDataset(
            base_dataset=dataset,
            variant=variant,
            seed=args.seed,
            knn_k=config["data"]["knn_k"],
            edge_feature_version=edge_feature_version,
            mut_aa_esm_mode=args.mut_aa_esm_mode,
        )
        max_nodes_per_batch = int(config["data"].get("eval_max_nodes_per_batch", config["data"].get("max_nodes_per_batch", 0)))
        if max_nodes_per_batch > 0:
            batch_sampler = LengthBucketBatchSampler(
                dataset=cf_dataset,
                max_nodes_per_batch=max_nodes_per_batch,
                bucket_size=int(config["data"].get("length_bucket_size", 256)),
                shuffle=False,
                seed=int(args.seed),
                max_graphs_per_batch=int(batch_size),
                drop_last=False,
            )
            loader = make_pyg_loader(
                cf_dataset,
                batch_sampler=batch_sampler,
                num_workers=int(config["data"].get("eval_num_workers", num_workers)),
                pin_memory=bool(config["data"].get("eval_pin_memory", False)),
                prefetch_factor=int(config["data"].get("eval_prefetch_factor", 1)),
                persistent_workers=False,
            )
        else:
            loader = DataLoader(cf_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        metrics, prediction_rows = evaluate_counterfactual_variant(model, loader, device, eval_cfg, variant)
        variant_metrics[variant] = metrics

        with (out_dir / f"metrics_{variant}.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        write_predictions(out_dir / f"predictions_{variant}.csv", prediction_rows)

    summary_rows = build_summary_rows(variant_metrics)
    summary_payload = {
        "split": args.split,
        "seed": args.seed,
        "mut_aa_esm_mode": args.mut_aa_esm_mode,
        "variants": VARIANTS,
        "metrics_by_variant": variant_metrics,
        "summary_rows": summary_rows,
    }
    with (out_dir / "counterfactual_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    write_summary_csv(out_dir / "counterfactual_summary.csv", summary_rows)

    print(f"Counterfactual summary written to: {out_dir / 'counterfactual_summary.csv'}")
    print()
    print("Expected interpretation:")
    print("- shuffle_site should worsen shell_mae and reduce AUPRC if mutation-site localization matters.")
    print("- shuffle_mutant_aa should reduce AUPRC if mutation identity matters.")
    print("- wt_wt_negative should produce near-zero displacement and mostly silent predictions.")
    print("- reverse_mutation should change predictions but does not need to be symmetric.")
    print("- remove_mutation_context should worsen metrics if mutation-conditioned FiLM/gating is useful.")


if __name__ == "__main__":
    main()
