from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.models import build_model
from scripts.counterfactual_tests import (
    MUT_SITE_COL,
    MUT_VEC_END,
    MUT_VEC_START,
    apply_remove_mutation_context,
    apply_reverse_mutation,
    apply_shuffle_mutant_aa,
    apply_shuffle_site,
    apply_wt_wt_negative,
    clone_data,
)


def make_data(num_nodes: int = 6) -> Data:
    data = Data()
    data.x_basic = torch.zeros(num_nodes, 57, dtype=torch.float32)
    data.x_basic[:, 0] = 1.0
    data.x_basic[2, MUT_SITE_COL] = 1.0
    mutation_vector = torch.zeros(20, dtype=torch.float32)
    mutation_vector[1] = 1.0
    mutation_vector[0] = -1.0
    data.x_basic[:, MUT_VEC_START:MUT_VEC_END] = mutation_vector[None, :]
    data.x_basic[:, 41:57] = torch.randn(num_nodes, 16)

    data.pos = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0], [9.0, 0.0, 0.0], [12.0, 0.0, 0.0], [15.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    edge_index = []
    for i in range(num_nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    data.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data.edge_attr = torch.randn(data.edge_index.size(1), 68)
    data.radii = torch.tensor([6.0, 3.0, 0.0, 3.0, 6.0, 9.0], dtype=torch.float32)
    data.shell_id = torch.tensor([1, 0, 0, 0, 1, 2], dtype=torch.long)
    data.is_mutation_site = data.x_basic[:, MUT_SITE_COL].clone()
    data.mut_pos = torch.tensor([2], dtype=torch.long)
    data.esm_wt = torch.randn(num_nodes, 8)
    data.esm_delta = torch.randn(num_nodes, 8)
    data.y_disp = torch.rand(num_nodes)
    data.y_perturbed = torch.randint(0, 2, (num_nodes,), dtype=torch.float32)
    data.y_radius = torch.tensor([9.0], dtype=torch.float32)
    data.y_class = torch.tensor([2], dtype=torch.long)
    data.sample_id = "sample_a"
    data.cluster_id_30 = "cluster_a"
    data.wt_aa = "A"
    data.mut_aa = "C"
    data.mutation_key = "A->C"
    return data


def test_shuffle_site_changes_single_site_indicator_and_graph_fields() -> None:
    data = apply_shuffle_site(clone_data(make_data()), seed=42, knn_k=4, edge_feature_version="v4")
    mutation_site_indices = torch.nonzero(data.x_basic[:, MUT_SITE_COL] > 0.5, as_tuple=False).view(-1)
    assert mutation_site_indices.numel() == 1
    assert int(mutation_site_indices.item()) != 2
    assert not torch.equal(data.radii, torch.tensor([6.0, 3.0, 0.0, 3.0, 6.0, 9.0], dtype=torch.float32))
    assert not torch.equal(data.shell_id, torch.tensor([1, 0, 0, 0, 1, 2], dtype=torch.long))


def test_shuffle_mutant_aa_changes_mutation_vector_columns() -> None:
    original = make_data()
    data = apply_shuffle_mutant_aa(clone_data(original), seed=42, mut_aa_esm_mode="reuse_delta")
    assert not torch.equal(data.x_basic[:, MUT_VEC_START:MUT_VEC_END], original.x_basic[:, MUT_VEC_START:MUT_VEC_END])


def test_wt_wt_negative_zeros_mutation_vector_and_esm_delta() -> None:
    data = apply_wt_wt_negative(clone_data(make_data()))
    assert torch.equal(data.x_basic[:, MUT_VEC_START:MUT_VEC_END], torch.zeros_like(data.x_basic[:, MUT_VEC_START:MUT_VEC_END]))
    assert torch.equal(data.esm_delta, torch.zeros_like(data.esm_delta))


def test_reverse_mutation_negates_mutation_vector_and_esm_delta() -> None:
    original = make_data()
    data = apply_reverse_mutation(clone_data(original))
    assert torch.equal(data.x_basic[:, MUT_VEC_START:MUT_VEC_END], -original.x_basic[:, MUT_VEC_START:MUT_VEC_END])
    assert torch.equal(data.esm_delta, -original.esm_delta)


def test_remove_mutation_context_does_not_change_input_data() -> None:
    original = make_data()
    data = apply_remove_mutation_context(clone_data(original))
    for key in ["x_basic", "esm_delta", "radii", "shell_id", "edge_index", "edge_attr"]:
        assert torch.equal(getattr(data, key), getattr(original, key))


def test_base_v5_forward_runs_with_disable_mutation_context() -> None:
    data = make_data()
    model = build_model(
        "base_v5",
        {
            "esm_dim": 8,
            "basic_dim": 57,
            "edge_dim": 68,
            "esm_proj_dim": 4,
            "hidden_dim": 16,
            "num_layers": 2,
            "dropout": 0.1,
            "num_classes": 3,
        },
    )
    kwargs = {
        "x_basic": data.x_basic,
        "esm_wt": data.esm_wt,
        "esm_delta": data.esm_delta,
        "edge_index": data.edge_index,
        "edge_attr": data.edge_attr,
        "shell_id": data.shell_id,
        "batch": torch.zeros(data.x_basic.size(0), dtype=torch.long),
        "radii": data.radii,
        "is_mutation_site": data.is_mutation_site,
        "mut_pos": data.mut_pos,
    }
    outputs_normal = model(**kwargs)
    outputs_disabled = model(**kwargs, disable_mutation_context=True)
    assert set(outputs_normal.keys()) == set(outputs_disabled.keys())

