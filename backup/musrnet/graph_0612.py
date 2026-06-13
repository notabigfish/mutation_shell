from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from musrnet.constants import RBF_CENTERS, RBF_SIGMA


def radial_basis(distances: np.ndarray) -> np.ndarray:
    centers = RBF_CENTERS[None, :]
    return np.exp(-((distances[:, None] - centers) ** 2) / (2.0 * (RBF_SIGMA**2))).astype(np.float32)


def knn_edges(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = coords.shape[0]
    if num_nodes <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    distance_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(distance_matrix, np.inf)
    k_eff = min(k, max(num_nodes - 1, 1))
    neighbors = np.argpartition(distance_matrix, kth=k_eff - 1, axis=1)[:, :k_eff]
    src = np.repeat(np.arange(num_nodes), k_eff)
    dst = neighbors.reshape(-1)
    distances = distance_matrix[src, dst].astype(np.float32)
    return np.stack([src, dst], axis=0), distances


def build_graph(sample: dict[str, Any], esm_data: dict[str, torch.Tensor], knn_k: int) -> Data:
    coords = sample["coords_wt"].cpu().numpy()  # seq_len, 3
    edge_index_np, edge_distances = knn_edges(coords, knn_k)
    seq_sep = np.minimum(np.abs(edge_index_np[0] - edge_index_np[1]), 64).astype(np.float32) / 64.0
    edge_attr_np = np.concatenate([radial_basis(edge_distances), seq_sep[:, None]], axis=1)
    data = Data()
    data.x_basic = sample["x_basic"].float()
    data.esm_wt = esm_data["esm_wt"].float()
    data.esm_delta = esm_data["esm_delta"].float()
    data.pos = sample["coords_wt"].float()
    data.edge_index = torch.from_numpy(edge_index_np).long()
    data.edge_attr = torch.from_numpy(edge_attr_np).float()
    data.y_disp = sample["displacement"].float()
    data.y_perturbed = sample["perturbed"].float()
    data.y_radius = sample["radius_label"].float()
    data.y_class = sample["class_label"].long()
    data.shell_id = sample["shell_id"].long()
    data.sample_id = sample["sample_id"]
    data.cluster_id_30 = sample["cluster_id_30"]
    return data
