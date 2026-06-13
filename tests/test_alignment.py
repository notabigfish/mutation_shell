from __future__ import annotations

import numpy as np

from musrnet.alignment import kabsch_align


def test_kabsch_alignment_recovers_rigid_transform() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 3)).astype(np.float32)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    t = rng.normal(size=(3,)).astype(np.float32)
    Y = X @ Q.T + t
    X_aligned, _, _ = kabsch_align(X, Y, np.ones(X.shape[0], dtype=bool))
    rmsd = np.sqrt(np.mean(np.sum((X_aligned - Y) ** 2, axis=1)))
    assert rmsd < 1e-5
