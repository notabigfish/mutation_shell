from __future__ import annotations

import torch
import torch.nn as nn


class ZeroResponseBaseline(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()

    def forward(self, x_basic: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        zeros = x_basic.new_zeros(x_basic.shape[0])
        return {
            "disp": zeros,
            "perturbed_logit": torch.full_like(zeros, -20.0),
        }
