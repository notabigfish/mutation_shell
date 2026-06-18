from __future__ import annotations

from typing import Any

from musrnet.models.base_v1 import MuSRNet as BaseV1MuSRNet
from musrnet.models.base_v2 import MuSRNet as BaseV2MuSRNet
from musrnet.models.base_v4 import MuSRNet as BaseV4MuSRNet
from musrnet.models.base_v5 import MuSRNet as BaseV5MuSRNet


MODEL_REGISTRY = {
    "base_v1": BaseV1MuSRNet,
    "base_v2": BaseV2MuSRNet,
    "base_v4": BaseV4MuSRNet,
    "base_v5": BaseV5MuSRNet,
}


def get_model_class(model_name: str):
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_name='{model_name}'. Available models: {available}")
    return MODEL_REGISTRY[model_name]


def build_model(model_name: str, model_kwargs: dict[str, Any]):
    model_class = get_model_class(model_name)
    return model_class(**model_kwargs)
