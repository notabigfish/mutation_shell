from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESMMLPBaseline(nn.Module):
    def __init__(
        self,
        esm_dim: int,
        basic_dim: int,
        hidden_dim: int = 256,
        esm_proj_dim: int = 128,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        self.proj_wt = nn.Linear(esm_dim, esm_proj_dim)
        self.proj_delta = nn.Linear(esm_dim, esm_proj_dim)
        input_dim = basic_dim + 2 * esm_proj_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.disp_head = nn.Linear(hidden_dim, 1)
        self.pert_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_basic: torch.Tensor, esm_wt: torch.Tensor, esm_delta: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        h = torch.cat([x_basic, self.proj_wt(esm_wt), self.proj_delta(esm_delta)], dim=-1)
        h = self.mlp(h)
        return {
            "disp": F.softplus(self.disp_head(h)).squeeze(-1),
            "perturbed_logit": self.pert_head(h).squeeze(-1),
        }
