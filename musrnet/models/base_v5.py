from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from musrnet.models.base_v4 import MutationFiLM, MuSRNetV4Layer


class MuSRNet(nn.Module):
    def __init__(
        self,
        esm_dim: int,
        basic_dim: int,
        edge_dim: int,
        esm_proj_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        del num_classes
        self.mutation_site_indicator_column = 20
        self.proj_wt = nn.Linear(esm_dim, esm_proj_dim)
        self.proj_delta = nn.Linear(esm_dim, esm_proj_dim)
        self.input_proj = nn.Linear(basic_dim + 2 * esm_proj_dim, hidden_dim)
        self.mutation_mlp = nn.Sequential(
            nn.Linear(basic_dim + 2 * esm_proj_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.film = MutationFiLM(hidden_dim=hidden_dim, num_rbf=16, dropout=dropout)
        self.layers = nn.ModuleList(
            [MuSRNetV4Layer(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.activation = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.pert_head = nn.Linear(hidden_dim, 1)
        self.background_head = nn.Linear(hidden_dim, 1)
        self.excess_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def _mutation_node_indices(
        self,
        x_basic: torch.Tensor,
        batch: torch.Tensor,
        is_mutation_site: torch.Tensor | None,
    ) -> torch.Tensor:
        if is_mutation_site is None:
            mutation_mask = x_basic[:, self.mutation_site_indicator_column] > 0.5
        else:
            mutation_mask = is_mutation_site > 0.5
        mutation_indices = torch.nonzero(mutation_mask, as_tuple=False).squeeze(-1)
        batch_size = int(batch.max().item()) + 1 if batch.numel() else 1
        if mutation_indices.numel() != batch_size:
            raise ValueError(
                f"Expected one mutation-site node per graph, found {mutation_indices.numel()} for batch size {batch_size}"
            )
        return mutation_indices

    def forward(
        self,
        x_basic: torch.Tensor,
        esm_wt: torch.Tensor,
        esm_delta: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        shell_id: torch.Tensor,
        batch: torch.Tensor | None = None,
        radii: torch.Tensor | None = None,
        is_mutation_site: torch.Tensor | None = None,
        mut_pos: torch.Tensor | None = None,
        disable_mutation_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        del shell_id, mut_pos
        if batch is None:
            batch = x_basic.new_zeros(x_basic.size(0), dtype=torch.long)
        if radii is None:
            raise ValueError("base_v5 requires per-node radii")

        proj_wt = self.proj_wt(esm_wt)
        proj_delta = self.proj_delta(esm_delta)
        node_input = torch.cat([x_basic, proj_wt, proj_delta], dim=-1)
        h = self.drop(self.activation(self.input_proj(node_input)))

        mutation_indices = self._mutation_node_indices(x_basic, batch, is_mutation_site)
        mutation_input = torch.cat(
            [x_basic[mutation_indices], proj_wt[mutation_indices], proj_delta[mutation_indices]],
            dim=-1,
        )
        mutation_context = self.mutation_mlp(mutation_input)
        if disable_mutation_context:
            mutation_context = torch.zeros_like(mutation_context)
        h = self.film(h, mutation_context[batch], radii)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, mutation_context, batch)

        perturbed_logit = self.pert_head(h).squeeze(-1)
        p = torch.sigmoid(perturbed_logit)
        background = F.softplus(self.background_head(h)).squeeze(-1)
        excess = F.softplus(self.excess_head(h)).squeeze(-1)
        disp = background + p * excess
        disp_logvar = self.logvar_head(h).squeeze(-1).clamp(-5.0, 3.0)
        return {
            "disp": disp,
            "background": background,
            "excess": excess,
            "perturbed_logit": perturbed_logit,
            "disp_logvar": disp_logvar,
        }
