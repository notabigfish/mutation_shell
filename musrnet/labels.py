from __future__ import annotations

from typing import Any

import numpy as np

from musrnet.alignment import kabsch_align
from musrnet.constants import ALIGNMENT_EXCLUSION_RADIUS, PERTURBATION_THRESHOLD, SHELL_BOUNDS


def compute_shell_ids(radii: np.ndarray) -> np.ndarray:
    shell_ids = np.full(radii.shape, 4, dtype=np.int64)
    shell_ids[radii <= SHELL_BOUNDS[3]] = 3
    shell_ids[radii <= SHELL_BOUNDS[2]] = 2
    shell_ids[radii <= SHELL_BOUNDS[1]] = 1
    shell_ids[radii <= SHELL_BOUNDS[0]] = 0
    return shell_ids


def build_structural_labels(
    coords_wt: np.ndarray,
    coords_mut: np.ndarray,
    mut_pos: int,
    displacement_threshold: float = PERTURBATION_THRESHOLD,
) -> dict[str, Any]:
    if not (0 <= mut_pos < coords_wt.shape[0]):
        raise ValueError("Mutation position is out of range")
    if coords_wt.shape != coords_mut.shape:
        raise ValueError("Coordinate arrays must have the same shape")

    mut_coord = coords_wt[mut_pos]
    radii = np.linalg.norm(coords_wt - mut_coord[None, :], axis=1)
    align_mask = radii > ALIGNMENT_EXCLUSION_RADIUS
    coords_wt_aligned, _, _ = kabsch_align(coords_wt, coords_mut, align_mask)
    displacement = np.linalg.norm(coords_wt_aligned - coords_mut, axis=1)
    shell_id = compute_shell_ids(radii)
    perturbed = (displacement > displacement_threshold).astype(np.float32)
    radius_label = float(radii[perturbed.astype(bool)].max()) if perturbed.any() else 0.0
    max_disp = float(displacement.max()) if displacement.size else 0.0
    if max_disp <= displacement_threshold:
        class_label = 0
    elif radius_label <= 8.0:
        class_label = 1
    else:
        class_label = 2
    # c1000 class distribution: [0: 44269, 1: 3501, 2: 182721]
    return {
        "coords_wt_aligned": coords_wt_aligned.astype(np.float32),
        "displacement": displacement.astype(np.float32),
        "radii": radii.astype(np.float32),
        "shell_id": shell_id,
        "perturbed": perturbed,
        "radius_label": np.array([radius_label], dtype=np.float32),
        "class_label": np.array([class_label], dtype=np.int64),
    }


def derive_class_from_pred(pred_disp_graph, pred_radius, displacement_threshold=PERTURBATION_THRESHOLD, radius_threshold=8.0):
    max_disp = float(pred_disp_graph.max().item()) if pred_disp_graph.numel() else 0.0
    radius_label = float(pred_radius.item())

    if max_disp <= displacement_threshold:
        return 0
    elif radius_label <= radius_threshold:
        return 1
    else:
        return 2