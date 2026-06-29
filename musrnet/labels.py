from __future__ import annotations

from typing import Any

import numpy as np

from musrnet.alignment import align_by_variant
from musrnet.constants import PERTURBATION_THRESHOLD, SHELL_BOUNDS


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
    alignment_variant: str = "kabsch_exclude_4A",
    tmalign_bin: str | None = None,
    sample_id: str | None = None,
) -> dict[str, Any]:
    if not (0 <= mut_pos < coords_wt.shape[0]):
        raise ValueError("Mutation position is out of range")
    if coords_wt.shape != coords_mut.shape:
        raise ValueError("Coordinate arrays must have the same shape")

    mut_coord = coords_wt[mut_pos]
    radii = np.linalg.norm(coords_wt - mut_coord[None, :], axis=1)
    alignment = align_by_variant(
        coords_wt=coords_wt,
        coords_mut=coords_mut,
        mut_pos=mut_pos,
        variant=alignment_variant,
        tmalign_bin=tmalign_bin,
        sample_id=sample_id,
    )
    coords_wt_aligned = alignment.coords_wt_aligned
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
        "alignment_variant": alignment.alignment_name,
        "alignment_rmsd": np.array([alignment.alignment_rmsd], dtype=np.float32),
        "alignment_n_residues": np.array([alignment.n_aligned_residues], dtype=np.int64),
        "alignment_mask_fraction": np.array([alignment.mask.mean()], dtype=np.float32),
        "alignment_metadata": alignment.metadata,
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
