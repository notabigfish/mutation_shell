from __future__ import annotations

import torch
import torch.nn as nn


class MutationTypeShellMeanBaseline(nn.Module):
    def __init__(self, min_count: int = 20, **_: object) -> None:
        super().__init__()
        self.min_count = int(min_count)
        self.global_mu = 0.0
        self.global_pi = 1e-6
        self.shell_mu = [0.0] * 5
        self.shell_pi = [1e-6] * 5
        self.stats_by_key: dict[tuple[str, int], dict[str, float]] = {}

    def set_stats(
        self,
        *,
        global_mu: float,
        global_pi: float,
        shell_mu: list[float],
        shell_pi: list[float],
        stats_by_key: dict[tuple[str, int], dict[str, float]],
    ) -> None:
        self.global_mu = float(global_mu)
        self.global_pi = min(max(float(global_pi), 1e-6), 1.0 - 1e-6)
        self.shell_mu = [float(v) for v in shell_mu]
        self.shell_pi = [min(max(float(v), 1e-6), 1.0 - 1e-6) for v in shell_pi]
        self.stats_by_key = stats_by_key

    def _predict_one(self, mutation_key: str, shell_idx: int) -> tuple[float, float]:
        stats = self.stats_by_key.get((mutation_key, int(shell_idx)))
        if stats is not None and int(stats["count"]) >= self.min_count:
            return float(stats["mu"]), float(stats["pi"])
        return self.shell_mu[int(shell_idx)], self.shell_pi[int(shell_idx)]

    def forward(self, shell_id: torch.Tensor, mutation_key: list[str] | tuple[str, ...], **_: torch.Tensor) -> dict[str, torch.Tensor]:
        disp_values = []
        prob_values = []
        for key, shell_idx in zip(mutation_key, shell_id.detach().cpu().tolist()):
            mu, pi = self._predict_one(str(key), int(shell_idx))
            disp_values.append(mu)
            prob_values.append(min(max(pi, 1e-6), 1.0 - 1e-6))
        disp = torch.tensor(disp_values, dtype=torch.float32, device=shell_id.device)
        prob = torch.tensor(prob_values, dtype=torch.float32, device=shell_id.device)
        logit = torch.log(prob / (1.0 - prob))
        return {
            "disp": disp,
            "perturbed_logit": logit,
        }
