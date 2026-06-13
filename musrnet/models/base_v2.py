from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import scatter

class MuSRNetLayer(nn.Module):
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
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        messages = self.edge_mlp(torch.cat([h[src], h[dst], edge_attr], dim=-1))
        messages = messages.to(h.dtype)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)

        degree = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        degree.index_add_(0, dst, torch.ones(dst.size(0), device=h.device, dtype=h.dtype))
        degree = degree.clamp_min(1.0).unsqueeze(-1)

        agg = agg / degree
        update = self.node_mlp(torch.cat([h, agg], dim=-1))
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
        self.proj_wt = nn.Linear(esm_dim, esm_proj_dim)
        self.proj_delta = nn.Linear(esm_dim, esm_proj_dim)
        self.input_proj = nn.Linear(basic_dim + esm_proj_dim + esm_proj_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [MuSRNetLayer(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.activation = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.disp_head = nn.Linear(hidden_dim, 1)
        self.pert_head = nn.Linear(hidden_dim, 1)
        shell_dim = hidden_dim * 5
        self.radius_head = nn.Sequential(
            nn.Linear(shell_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # self.class_head = nn.Sequential(
        #     nn.Linear(shell_dim, hidden_dim),
        #     nn.SiLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, num_classes),
        # )
        self.active_head = nn.Sequential(
            nn.Linear(shell_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.local_head = nn.Sequential(
            nn.Linear(shell_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_basic: torch.Tensor,
        esm_wt: torch.Tensor,
        esm_delta: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        shell_id: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if batch is None:
            batch = x_basic.new_zeros(x_basic.size(0), dtype=torch.long)
        q = torch.cat([x_basic, self.proj_wt(esm_wt), self.proj_delta(esm_delta)], dim=-1)
        h = self.drop(self.activation(self.input_proj(q)))
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        disp_pred = torch.nn.functional.softplus(self.disp_head(h)).squeeze(-1)
        perturbed_logit = self.pert_head(h).squeeze(-1)
        batch_size = int(batch.max().item()) + 1 if batch.numel() else 1
        combined_index = batch * 5 + shell_id
        shell_pooled = scatter(h, combined_index, dim=0, dim_size=batch_size * 5, reduce="mean")
        shell_tensor = shell_pooled.view(batch_size, -1)
        
        radius_pred = torch.nn.functional.softplus(self.radius_head(shell_tensor)).squeeze(-1)
        # class_logit = self.class_head(shell_tensor)
        active_logit = self.active_head(shell_tensor).squeeze(-1)
        local_logit = self.local_head(shell_tensor).squeeze(-1)
        return {
            "disp": disp_pred,
            "perturbed_logit": perturbed_logit,
            "radius": radius_pred,
            # "class_logit": class_logit,
            "active_logit": active_logit,
            "local_logit": local_logit,
        }
