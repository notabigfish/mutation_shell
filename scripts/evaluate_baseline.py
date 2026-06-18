from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Batch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.dataset import build_basic_features, create_or_load_splits, load_sample_from_manifest, load_samples_manifest
from musrnet.eval_utils import derive_radius_and_class_for_graph
from musrnet.graph import build_graph
from musrnet.metrics import compute_metrics
from musrnet.models import build_model
from musrnet.train_utils import load_yaml, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate non-trainable strict baselines")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def graph_from_sample(sample: dict[str, object]):
    sample = dict(sample)
    sample["x_basic"] = build_basic_features(sample)
    zeros = torch.zeros((sample["x_basic"].shape[0], 1), dtype=torch.float32)
    esm_data = {"esm_wt": zeros, "esm_delta": zeros}
    return build_graph(sample, esm_data, knn_k=1, edge_feature_version="v1")


def fit_global_stats(samples: list[dict[str, object]]) -> tuple[float, float]:
    disp_sum = 0.0
    pert_sum = 0.0
    count = 0
    for sample in samples:
        disp = sample["displacement"].float()
        pert = sample["perturbed"].float()
        disp_sum += float(disp.sum().item())
        pert_sum += float(pert.sum().item())
        count += int(disp.numel())
    mu = disp_sum / max(count, 1)
    pi = min(max(pert_sum / max(count, 1), 1e-6), 1.0 - 1e-6)
    return mu, pi


def fit_shell_stats(samples: list[dict[str, object]], global_mu: float, global_pi: float) -> tuple[list[float], list[float]]:
    disp_sum = [0.0] * 5
    pert_sum = [0.0] * 5
    count = [0] * 5
    for sample in samples:
        for k, d, p in zip(sample["shell_id"].tolist(), sample["displacement"].tolist(), sample["perturbed"].tolist()):
            disp_sum[int(k)] += float(d)
            pert_sum[int(k)] += float(p)
            count[int(k)] += 1
    shell_mu = [global_mu if count[k] == 0 else disp_sum[k] / count[k] for k in range(5)]
    shell_pi = [global_pi if count[k] == 0 else min(max(pert_sum[k] / count[k], 1e-6), 1.0 - 1e-6) for k in range(5)]
    return shell_mu, shell_pi


def fit_mutation_shell_stats(samples: list[dict[str, object]]) -> dict[tuple[str, int], dict[str, float]]:
    buckets: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: {"disp_sum": 0.0, "pert_sum": 0.0, "count": 0})
    for sample in samples:
        key = f"{sample['wt_aa']}->{sample['mut_aa']}"
        for shell_idx, disp_value, pert_value in zip(sample["shell_id"].tolist(), sample["displacement"].tolist(), sample["perturbed"].tolist()):
            bucket = buckets[(key, int(shell_idx))]
            bucket["disp_sum"] += float(disp_value)
            bucket["pert_sum"] += float(pert_value)
            bucket["count"] += 1
    stats = {}
    for key, value in buckets.items():
        count = max(int(value["count"]), 1)
        stats[key] = {
            "mu": float(value["disp_sum"] / count),
            "pi": float(min(max(value["pert_sum"] / count, 1e-6), 1.0 - 1e-6)),
            "count": int(value["count"]),
        }
    return stats


def fit_baseline(model, baseline_cfg: dict[str, object], train_samples: list[dict[str, object]]) -> dict[str, object]:
    baseline_type = baseline_cfg["type"]
    if baseline_type == "zero_response":
        return {"type": baseline_type}
    global_mu, global_pi = fit_global_stats(train_samples)
    if baseline_type == "global_mean":
        model.set_stats(global_mu, global_pi)
        return {"type": baseline_type, "global_mu": global_mu, "global_pi": global_pi}
    shell_mu, shell_pi = fit_shell_stats(train_samples, global_mu, global_pi)
    if baseline_type == "shell_mean":
        model.set_stats(shell_mu, shell_pi)
        return {"type": baseline_type, "global_mu": global_mu, "global_pi": global_pi, "shell_mu": shell_mu, "shell_pi": shell_pi}
    if baseline_type == "mutation_type_shell_mean":
        stats_by_key = fit_mutation_shell_stats(train_samples)
        model.set_stats(global_mu=global_mu, global_pi=global_pi, shell_mu=shell_mu, shell_pi=shell_pi, stats_by_key=stats_by_key)
        serializable = {f"{key[0]}|{key[1]}": value for key, value in stats_by_key.items()}
        return {
            "type": baseline_type,
            "global_mu": global_mu,
            "global_pi": global_pi,
            "shell_mu": shell_mu,
            "shell_pi": shell_pi,
            "min_count": int(baseline_cfg.get("min_count", 20)),
            "mutation_shell_stats": serializable,
        }
    raise ValueError(f"Unsupported baseline type: {baseline_type}")


@torch.no_grad()
def evaluate_split(model, samples: list[dict[str, object]], eval_cfg: dict[str, object]) -> tuple[dict[str, float], list[dict[str, object]]]:
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
    rows: list[dict[str, object]] = []
    for sample in tqdm(samples, desc="Evaluating baseline"):
        graph = graph_from_sample(sample)
        batch = Batch.from_data_list([graph])
        outputs = model(
            x_basic=batch.x_basic,
            esm_wt=batch.esm_wt,
            esm_delta=batch.esm_delta,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            shell_id=batch.shell_id,
            batch=batch.batch,
            mutation_key=[graph.mutation_key] * graph.x_basic.shape[0],
        )
        disp = outputs["disp"].detach().cpu()
        probs = torch.sigmoid(outputs["perturbed_logit"]).detach().cpu()
        radii = graph.radii.detach().cpu()
        pred_radius, pred_class = derive_radius_and_class_for_graph(
            pred_disp=disp,
            pred_perturbed_prob=probs,
            radii=radii,
            displacement_threshold=float(eval_cfg.get("displacement_threshold", 1.0)),
            response_threshold=float(eval_cfg.get("response_threshold", 0.5)),
            radius_threshold=float(eval_cfg.get("radius_threshold", 8.0)),
        )
        true_radius = float(graph.y_radius.view(-1)[0].item())
        true_class = int(graph.y_class.view(-1)[0].item())
        nodes = graph.x_basic.shape[0]
        records["true_disp"].extend(graph.y_disp.tolist())
        records["pred_disp"].extend(disp.tolist())
        records["shell_id"].extend(graph.shell_id.tolist())
        records["true_perturbed"].extend(graph.y_perturbed.tolist())
        records["pred_perturbed_prob"].extend(probs.tolist())
        records["true_radius"].extend([true_radius] * nodes)
        records["pred_radius"].extend([pred_radius] * nodes)
        records["true_class"].extend([true_class] * nodes)
        records["pred_class"].extend([pred_class] * nodes)
        records["cluster_id_30"].extend([graph.cluster_id_30] * nodes)
        for residue_index in range(nodes):
            rows.append(
                {
                    "sample_id": graph.sample_id,
                    "cluster_id_30": graph.cluster_id_30,
                    "residue_index": residue_index,
                    "shell_id": int(graph.shell_id[residue_index].item()),
                    "radii": float(graph.radii[residue_index].item()),
                    "true_displacement": float(graph.y_disp[residue_index].item()),
                    "pred_displacement": float(disp[residue_index].item()),
                    "true_perturbed": float(graph.y_perturbed[residue_index].item()),
                    "pred_perturbed_prob": float(probs[residue_index].item()),
                    "true_radius": true_radius,
                    "pred_radius": pred_radius,
                    "true_class": true_class,
                    "pred_class": pred_class,
                }
            )
    return compute_metrics(records), rows


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
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
    manifest = load_samples_manifest(config["paths"]["samples"])
    cluster_pkl_path = PROJECT_ROOT / "data" / "SingleMutPairs2024_cluster30.pkl"
    splits = create_or_load_splits(manifest, config["paths"]["splits"], cluster_pkl_path, config["seed"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(config["model_name"], config.get("model", {}))

    train_samples = [load_sample_from_manifest(manifest, sample_id) for sample_id in splits["train"]]
    stats = fit_baseline(model, config["baseline"], train_samples)
    with (output_dir / "baseline_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    save_yaml(output_dir / "config_used.yaml", config)

    for split in ["train", "valid", "test"]:
        samples = [load_sample_from_manifest(manifest, sample_id) for sample_id in splits[split]]
        metrics, rows = evaluate_split(model, samples, config.get("eval", {}))
        with (output_dir / f"eval_{split}.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        write_predictions(output_dir / f"predictions_{split}.csv", rows)


if __name__ == "__main__":
    main()
