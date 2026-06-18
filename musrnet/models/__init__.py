from __future__ import annotations

from typing import Any

from musrnet.models.base_v1 import MuSRNet as BaseV1MuSRNet
from musrnet.models.base_v2 import MuSRNet as BaseV2MuSRNet
from musrnet.models.base_v4 import MuSRNet as BaseV4MuSRNet
from musrnet.models.base_v5 import MuSRNet as BaseV5MuSRNet
from musrnet.models.coordinate_residual import CoordinateResidualBaseline
from musrnet.models.esm_mlp import ESMMLPBaseline
from musrnet.models.geometry_gnn import GeometryGNNBaseline
from musrnet.models.global_mean import GlobalMeanBaseline
from musrnet.models.mutation_type_shell_mean import MutationTypeShellMeanBaseline
from musrnet.models.shell_mean import ShellMeanBaseline
from musrnet.models.zero_response import ZeroResponseBaseline


MODEL_REGISTRY = {
    "base_v1": BaseV1MuSRNet,
    "base_v2": BaseV2MuSRNet,
    "base_v4": BaseV4MuSRNet,
    "base_v5": BaseV5MuSRNet,
    "zero_response": ZeroResponseBaseline,
    "global_mean": GlobalMeanBaseline,
    "shell_mean": ShellMeanBaseline,
    "mutation_type_shell_mean": MutationTypeShellMeanBaseline,
    "esm_mlp": ESMMLPBaseline,
    "geometry_gnn": GeometryGNNBaseline,
    "coordinate_residual": CoordinateResidualBaseline,
}


def get_model_class(model_name: str):
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_name='{model_name}'. Available models: {available}")
    return MODEL_REGISTRY[model_name]


def build_model(model_name: str, model_kwargs: dict[str, Any]):
    model_class = get_model_class(model_name)
    return model_class(**model_kwargs)
