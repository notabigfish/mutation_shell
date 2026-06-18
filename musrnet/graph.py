from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from musrnet.constants import RBF_CENTERS, RBF_SIGMA


def radial_basis(distances: np.ndarray) -> np.ndarray:
    centers = RBF_CENTERS[None, :]
    return np.exp(-((distances[:, None] - centers) ** 2) / (2.0 * (RBF_SIGMA**2))).astype(np.float32)


def knn_edges(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = coords.shape[0]
    if num_nodes <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    k_eff = min(k, num_nodes - 1)
    tree = cKDTree(coords)
    distances, neighbors = tree.query(coords, k=k_eff + 1)
    distances = distances[:, 1:].astype(np.float32).reshape(-1)
    neighbors = neighbors[:, 1:].reshape(-1)
    dst = np.repeat(np.arange(num_nodes), k_eff)
    src = neighbors
    return np.stack([src, dst], axis=0), distances


def build_edge_attr_v1(edge_index_np: np.ndarray, edge_distances: np.ndarray) -> np.ndarray:
    seq_sep = np.minimum(np.abs(edge_index_np[0] - edge_index_np[1]), 64).astype(np.float32) / 64.0
    return np.concatenate([radial_basis(edge_distances), seq_sep[:, None]], axis=1)


def build_edge_attr_v4(
    coords: np.ndarray,
    edge_index_np: np.ndarray,
    edge_distances: np.ndarray,
    mut_pos: int,
    shell_id: np.ndarray,
) -> np.ndarray:
    src = edge_index_np[0]
    dst = edge_index_np[1]
    x_src = coords[src]
    x_dst = coords[dst]
    x_mut = coords[mut_pos]
    seq_sep = np.minimum(np.abs(src - dst), 64).astype(np.float32) / 64.0
    r_src = np.linalg.norm(x_src - x_mut[None, :], axis=1).astype(np.float32)
    r_dst = np.linalg.norm(x_dst - x_mut[None, :], axis=1).astype(np.float32)
    delta_r = (r_src - r_dst).astype(np.float32)
    v_edge = x_src - x_dst
    v_mut = x_mut[None, :] - x_dst
    denom = (np.linalg.norm(v_edge, axis=1) * np.linalg.norm(v_mut, axis=1) + 1e-8).astype(np.float32)
    cos_theta = (np.sum(v_edge * v_mut, axis=1) / denom).astype(np.float32)
    same_shell = (shell_id[src] == shell_id[dst]).astype(np.float32)
    return np.concatenate(
        [
            radial_basis(edge_distances),
            seq_sep[:, None],
            radial_basis(r_src),
            radial_basis(r_dst),
            radial_basis(np.abs(delta_r)),
            np.sign(delta_r)[:, None].astype(np.float32),
            cos_theta[:, None],
            same_shell[:, None],
        ],
        axis=1,
    )


def build_graph(
    sample: dict[str, Any],
    esm_data: dict[str, torch.Tensor],
    knn_k: int,
    edge_feature_version: str = "v1",
) -> Data:
    coords = sample["coords_wt"].cpu().numpy()
    edge_index_np, edge_distances = knn_edges(coords, knn_k)
    shell_id_np = sample["shell_id"].cpu().numpy()
    if edge_feature_version == "v4":
        edge_attr_np = build_edge_attr_v4(
            coords=coords,
            edge_index_np=edge_index_np,
            edge_distances=edge_distances,
            mut_pos=int(sample["mut_pos"]),
            shell_id=shell_id_np,
        )
    else:
        edge_attr_np = build_edge_attr_v1(edge_index_np, edge_distances)

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
    data.y_coord_residual = (sample["coords_mut"] - sample["coords_wt_aligned"]).float()
    data.shell_id = sample["shell_id"].long()
    data.radii = sample["radii"].float()
    data.is_mutation_site = sample["x_basic"][:, 20].float()
    data.mut_pos = torch.tensor([int(sample["mut_pos"])], dtype=torch.long)
    data.sample_id = sample["sample_id"]
    data.cluster_id_30 = sample["cluster_id_30"]
    data.wt_aa = sample["wt_aa"]
    data.mut_aa = sample["mut_aa"]
    data.mutation_key = f"{sample['wt_aa']}->{sample['mut_aa']}"
    return data
