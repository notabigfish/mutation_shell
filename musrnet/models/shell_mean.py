from __future__ import annotations

import torch
import torch.nn as nn


class ShellMeanBaseline(nn.Module):
    def __init__(self, shell_mu: list[float] | None = None, shell_pi: list[float] | None = None, **_: object) -> None:
        super().__init__()
        self.register_buffer("shell_mu", torch.tensor(shell_mu or [0.0] * 5, dtype=torch.float32))
        self.register_buffer("shell_pi", torch.tensor(shell_pi or [1e-6] * 5, dtype=torch.float32))

    def set_stats(self, shell_mu: list[float], shell_pi: list[float]) -> None:
        pi = [min(max(float(value), 1e-6), 1.0 - 1e-6) for value in shell_pi]
        self.shell_mu.copy_(torch.tensor(shell_mu, dtype=torch.float32, device=self.shell_mu.device))
        self.shell_pi.copy_(torch.tensor(pi, dtype=torch.float32, device=self.shell_pi.device))

    def forward(self, shell_id: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        disp = self.shell_mu[shell_id]
        pi = self.shell_pi[shell_id]
        logit = torch.log(pi / (1.0 - pi))
        return {
            "disp": disp,
            "perturbed_logit": logit,
        }
