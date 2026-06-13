from __future__ import annotations

import numpy as np

from musrnet.labels import build_structural_labels


def test_structural_labels_shapes_and_classes() -> None:
    L = 12
    coords_wt = np.stack([np.array([float(i) * 3.0, 0.0, 0.0], dtype=np.float32) for i in range(L)], axis=0)
    coords_mut = coords_wt.copy()
    coords_mut[4] += np.array([0.0, 1.5, 0.0], dtype=np.float32)
    coords_mut[5] += np.array([0.0, 1.2, 0.0], dtype=np.float32)
    labels = build_structural_labels(coords_wt, coords_mut, mut_pos=4)
    assert labels["displacement"].shape == (L,)
    assert set(labels["shell_id"].tolist()).issubset({0, 1, 2, 3, 4})
    assert np.isclose(labels["radius_label"][0], 3.0, atol=1e-4)
    assert labels["class_label"][0] == 1
