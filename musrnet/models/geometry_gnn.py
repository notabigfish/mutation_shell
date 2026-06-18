from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        sender, receiver = edge_index
        edge_input = torch.cat([h[receiver], h[sender], edge_attr.to(h.dtype)], dim=-1)
        messages = self.edge_mlp(edge_input).to(dtype=h.dtype)
        agg = torch.zeros_like(h)
        agg.index_add_(0, receiver, messages)
        degree = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        degree.index_add_(0, receiver, torch.ones(receiver.size(0), device=h.device, dtype=h.dtype))
        agg = agg / degree.clamp_min(1.0).unsqueeze(-1)
        update = self.node_mlp(torch.cat([h, agg], dim=-1)).to(dtype=h.dtype)
        return self.norm(h + self.res_scale * update)


class GeometryGNNBaseline(nn.Module):
    def __init__(self, basic_dim: int, edge_dim: int, hidden_dim: int = 256, num_layers: int = 6, dropout: float = 0.1, **_: object) -> None:
        super().__init__()
        self.input_proj = nn.Linear(basic_dim, hidden_dim)
        self.layers = nn.ModuleList([GeometryGNNLayer(hidden_dim, edge_dim, dropout) for _ in range(num_layers)])
        self.disp_head = nn.Linear(hidden_dim, 1)
        self.pert_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_basic: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input_proj(x_basic)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return {
            "disp": F.softplus(self.disp_head(h)).squeeze(-1),
            "perturbed_logit": self.pert_head(h).squeeze(-1),
        }
