from __future__ import annotations

import numpy as np


def kabsch_align(X: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("Coordinates must have shape [L, 3]")
    if mask.shape[0] != X.shape[0]:
        raise ValueError("Mask length must equal number of residues")

    mask = mask.astype(bool)
    if mask.sum() < 3:
        raise ValueError("At least three residues are required for Kabsch alignment")

    XA = X[mask]
    YA = Y[mask]
    cx = XA.mean(axis=0)
    cy = YA.mean(axis=0)
    X0 = XA - cx
    Y0 = YA - cy
    H = X0.T @ Y0
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1.0
        R = V @ U.T
    t = cy - R @ cx
    X_aligned = X @ R.T + t
    return X_aligned.astype(np.float32), R.astype(np.float32), t.astype(np.float32)
