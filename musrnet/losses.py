from __future__ import annotations

import torch
import torch.nn.functional as F


def huber(values: torch.Tensor, delta: float) -> torch.Tensor:
    abs_values = values.abs()
    quadratic = torch.minimum(abs_values, torch.tensor(delta, device=values.device, dtype=values.dtype))
    linear = abs_values - quadratic
    return 0.5 * quadratic**2 + delta * linear


def shell_balanced_disp_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    shell_id: torch.Tensor,
    batch: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    batch_size = int(batch.max().item()) + 1 if batch.numel() else 1
    losses = []
    per_residue = huber(pred - target, delta)
    for graph_idx in range(batch_size):
        graph_mask = batch == graph_idx
        shell_losses = []
        for shell_idx in range(5):
            mask = graph_mask & (shell_id == shell_idx)
            if mask.any():
                shell_losses.append(per_residue[mask].mean())
        if shell_losses:
            losses.append(torch.stack(shell_losses).mean())
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def shell_balanced_reduce(
    per_residue: torch.Tensor,
    shell_id: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(batch.max().item()) + 1 if batch.numel() else 1
    losses = []
    for graph_idx in range(batch_size):
        graph_mask = batch == graph_idx
        shell_losses = []
        for shell_idx in range(5):
            mask = graph_mask & (shell_id == shell_idx)
            if mask.any():
                shell_losses.append(per_residue[mask].mean())
        if shell_losses:
            losses.append(torch.stack(shell_losses).mean())
    if not losses:
        return per_residue.new_tensor(0.0)
    return torch.stack(losses).mean()

def _epoch_enabled_weight(
    base_weight: float,
    current_epoch: float,
    warmup: float,
    start_epoch: float | None,
) -> float:
    if start_epoch is None: return base_weight
    if current_epoch < start_epoch:
        return 0.0
    if current_epoch >= start_epoch + warmup:
        return base_weight
    return base_weight * (current_epoch - start_epoch + 1) / warmup

def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch,
    delta: float = 1.0,
    w_perturbed: float = 0.2,
    w_radius: float = 0.2,
    w_class: float = 0.2,
    current_epoch: float = 0.0,
    warmup: float = 5.0,
    start_perturbed_epoch: float | None = None,
    start_radius_epoch: float | None = None,
    start_class_epoch: float | None = None,
    use_log_disp: bool = False,
    use_heteroscedastic_disp: bool = False,
) -> dict[str, torch.Tensor]:
    if use_log_disp and use_heteroscedastic_disp and "disp_logvar" in outputs:
        target = torch.log1p(batch.y_disp)
        pred = torch.log1p(outputs["disp"])
        err2 = (pred - target) ** 2
        logvar = outputs["disp_logvar"]
        per_residue = 0.5 * torch.exp(-logvar) * err2 + 0.5 * logvar
        disp_loss = shell_balanced_reduce(per_residue, batch.shell_id, batch.batch)
    else:
        disp_loss = shell_balanced_disp_loss(outputs["disp"], batch.y_disp, batch.shell_id, batch.batch, delta)
    pert_loss = F.binary_cross_entropy_with_logits(outputs["perturbed_logit"], batch.y_perturbed)
    radius_loss = (
        torch.abs(outputs["radius"] - batch.y_radius.view(-1)).mean()
        if "radius" in outputs
        else disp_loss.new_tensor(0.0)
    )

    w_perturbed_eff = _epoch_enabled_weight(w_perturbed, current_epoch, warmup, start_perturbed_epoch)
    w_radius_eff = _epoch_enabled_weight(w_radius, current_epoch, warmup, start_radius_epoch)
    w_class_eff = _epoch_enabled_weight(w_class, current_epoch, warmup, start_class_epoch)

    total = disp_loss + w_perturbed_eff * pert_loss + w_radius_eff * radius_loss

    loss_dict = {
        "loss": total,
        "disp_loss": disp_loss,
        "pert_loss": pert_loss,
        "w_perturbed_eff": torch.tensor(w_perturbed_eff, device=disp_loss.device),
        "w_radius_eff": torch.tensor(w_radius_eff, device=disp_loss.device),
        "w_class_eff": torch.tensor(w_class_eff, device=disp_loss.device),
    }
    if "radius" in outputs:
        loss_dict["radius_loss"] = radius_loss
    return loss_dict
