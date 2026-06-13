from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from musrnet.models import build_model


def make_graph(num_nodes: int, esm_dim: int) -> Data:
    edge_index = []
    for i in range(num_nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data()
    data.x_basic = torch.randn(num_nodes, 57)
    data.esm_wt = torch.randn(num_nodes, esm_dim)
    data.esm_delta = torch.randn(num_nodes, esm_dim)
    data.edge_index = edge_index
    data.edge_attr = torch.randn(edge_index.size(1), 17)
    data.shell_id = torch.randint(0, 5, (num_nodes,), dtype=torch.long)
    data.y_disp = torch.rand(num_nodes)
    data.y_perturbed = torch.randint(0, 2, (num_nodes,), dtype=torch.float32)
    data.y_radius = torch.rand(1)
    data.y_class = torch.randint(0, 3, (1,), dtype=torch.long)
    data.sample_id = f"sample_{num_nodes}"
    data.cluster_id_30 = f"cluster_{num_nodes}"
    return data


def test_model_forward_shapes() -> None:
    esm_dim = 16
    batch = Batch.from_data_list([make_graph(5, esm_dim), make_graph(7, esm_dim)])
    model = build_model(
        "base_v1",
        {
            "esm_dim": esm_dim,
            "basic_dim": 57,
            "edge_dim": 17,
            "esm_proj_dim": 8,
            "hidden_dim": 32,
            "num_layers": 2,
            "dropout": 0.1,
            "num_classes": 3,
        },
    )
    outputs = model(
        x_basic=batch.x_basic,
        esm_wt=batch.esm_wt,
        esm_delta=batch.esm_delta,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        shell_id=batch.shell_id,
        batch=batch.batch,
    )
    assert outputs["disp"].shape == (12,)
    assert outputs["perturbed_logit"].shape == (12,)
    assert outputs["radius"].shape == (2,)
    assert outputs["class_logit"].shape == (2, 3)
