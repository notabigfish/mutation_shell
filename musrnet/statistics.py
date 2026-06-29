from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    differences: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    diffs = np.asarray(differences, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, diffs.size, size=diffs.size)
        means[i] = diffs[idx].mean()
    alpha = 1.0 - ci
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return low, high


def wilcoxon_signed_rank_pvalue(differences: np.ndarray) -> float:
    diffs = np.asarray(differences, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs != 0.0]
    if diffs.size < 2:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return float("nan")
    try:
        return float(wilcoxon(diffs).pvalue)
    except Exception:
        return float("nan")
