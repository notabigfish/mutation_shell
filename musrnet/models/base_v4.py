from __future__ import annotations

import torch
from torch import nn


class RBF(nn.Module):
    def __init__(self, num_rbf: int = 16, r_min: float = 0.0, r_max: float = 32.0, sigma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("centers", torch.linspace(r_min, r_max, num_rbf))
        self.sigma = float(sigma)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-(diff**2) / (2.0 * (self.sigma**2)))


class MutationFiLM(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.rbf = RBF(num_rbf=num_rbf)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

    def forward(self, h: torch.Tensor, mutation_context: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        film_input = torch.cat([h, mutation_context, self.rbf(radii)], dim=-1)
        gamma, beta = self.mlp(film_input).chunk(2, dim=-1)
        return h * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta


class MuSRNetV4Layer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        edge_input_dim = 3 * hidden_dim + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        mutation_context: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        sender, receiver = edge_index
        edge_context = mutation_context[batch[receiver]]
        edge_input = torch.cat([h[receiver], h[sender], edge_attr, edge_context], dim=-1)
        raw_message = self.edge_mlp(edge_input)
        gate = torch.sigmoid(self.gate_mlp(edge_input))
        message = (gate * raw_message).to(dtype=h.dtype)

        agg = torch.zeros_like(h)
        agg.index_add_(0, receiver, message)

        degree = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        degree.index_add_(0, receiver, torch.ones(receiver.size(0), device=h.device, dtype=h.dtype))
        agg = agg / degree.clamp_min(1.0).unsqueeze(-1)

        node_context = mutation_context[batch]
        update = self.node_mlp(torch.cat([h, agg, node_context], dim=-1)).to(dtype=h.dtype)
        return self.norm(h + self.res_scale * update)


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
        self.mag_head = nn.Linear(hidden_dim, 1)
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
    ) -> dict[str, torch.Tensor]:
        del shell_id, mut_pos
        if batch is None:
            batch = x_basic.new_zeros(x_basic.size(0), dtype=torch.long)
        if radii is None:
            raise ValueError("base_v4 requires per-node radii")

        proj_wt = self.proj_wt(esm_wt)
        proj_delta = self.proj_delta(esm_delta)
        node_input = torch.cat([x_basic, proj_wt, proj_delta], dim=-1)
        h = self.drop(self.activation(self.input_proj(node_input)))

        mutation_indices = self._mutation_node_indices(x_basic, batch, is_mutation_site)
        mutation_input = torch.cat([x_basic[mutation_indices], proj_wt[mutation_indices], proj_delta[mutation_indices]], dim=-1)
        mutation_context = self.mutation_mlp(mutation_input)
        h = self.film(h, mutation_context[batch], radii)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, mutation_context, batch)

        perturbed_logit = self.pert_head(h).squeeze(-1)
        magnitude = torch.nn.functional.softplus(self.mag_head(h)).squeeze(-1)
        disp = torch.sigmoid(perturbed_logit) * magnitude
        disp_logvar = torch.clamp(self.logvar_head(h).squeeze(-1), min=-5.0, max=3.0)
        return {
            "disp": disp,
            "perturbed_logit": perturbed_logit,
            "magnitude": magnitude,
            "disp_logvar": disp_logvar,
        }
