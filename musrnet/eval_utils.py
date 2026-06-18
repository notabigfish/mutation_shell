from __future__ import annotations

import torch


def derive_radius_and_class_for_graph(
    pred_disp: torch.Tensor,
    pred_perturbed_prob: torch.Tensor,
    radii: torch.Tensor,
    displacement_threshold: float = 1.0,
    response_threshold: float = 0.5,
    radius_threshold: float = 8.0,
) -> tuple[float, int]:
    score = pred_perturbed_prob * pred_disp
    pred_perturbed = score > response_threshold

    if pred_perturbed.any():
        pred_radius = float(radii[pred_perturbed].max().item())
    else:
        pred_radius = 0.0

    max_disp = float(pred_disp.max().item()) if pred_disp.numel() else 0.0
    if max_disp <= displacement_threshold:
        pred_class = 0
    elif pred_radius <= radius_threshold:
        pred_class = 1
    else:
        pred_class = 2
    return pred_radius, pred_class
