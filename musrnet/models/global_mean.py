from __future__ import annotations

import torch
import torch.nn as nn


class GlobalMeanBaseline(nn.Module):
    def __init__(self, mu: float = 0.0, pi: float = 1e-6, **_: object) -> None:
        super().__init__()
        self.register_buffer("mu", torch.tensor(float(mu), dtype=torch.float32))
        self.register_buffer("pi", torch.tensor(float(pi), dtype=torch.float32))

    def set_stats(self, mu: float, pi: float) -> None:
        pi = min(max(float(pi), 1e-6), 1.0 - 1e-6)
        self.mu.fill_(float(mu))
        self.pi.fill_(pi)

    def forward(self, x_basic: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        n = x_basic.shape[0]
        disp = self.mu.expand(n)
        logit = torch.log(self.pi / (1.0 - self.pi)).expand(n)
        return {
            "disp": disp,
            "perturbed_logit": logit,
        }
