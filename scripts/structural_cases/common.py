"""Small, shared numerical definitions for MuSRBench structural-case analysis."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

OUT = Path('results/structural_cases')
PRED = Path('outputs/call')
MODEL_PATHS = {
    'musrnet': PRED / 'base_v5_seed42/predictions_test.csv',
    'geometry': PRED / 'geometry_gnn/predictions_test.csv',
    'coordinate': PRED / 'coordinate_residual/predictions_test.csv',
    'global_loss': PRED / 'base_v5_global_loss_seed42/predictions_test.csv',
    'direct': PRED / 'base_v5_direct_seed42/predictions_test.csv',
    'pure_hurdle': PRED / 'base_v5_pure_hurdle_seed42/predictions_test.csv',
}


def sample_groups(path: Path, chunksize: int = 250000):
    """Stream contiguous per-sample rows, including across CSV chunk boundaries."""
    carry = None
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype={'sample_id': str, 'cluster_id_30': str}):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        sid = chunk['sample_id'].iat[-1]
        carry = chunk.loc[chunk['sample_id'] == sid].copy()
        complete = chunk.loc[chunk['sample_id'] != sid]
        for sample_id, group in complete.groupby('sample_id', sort=False):
            yield str(sample_id), group.sort_values('residue_index')
    if carry is not None:
        yield str(carry['sample_id'].iat[0]), carry.sort_values('residue_index')


def shell_mae(truth: np.ndarray, pred: np.ndarray, shell: np.ndarray) -> float:
    return float(np.mean([np.abs(truth[shell == k] - pred[shell == k]).mean()
                          for k in range(5) if np.any(shell == k)]))


def auprc(truth: np.ndarray, score: np.ndarray) -> float:
    z = truth > 1.0
    return float(average_precision_score(z, score)) if np.unique(z).size == 2 else float('nan')


def precision_at5(truth: np.ndarray, prob: np.ndarray) -> float:
    k = max(1, int(np.ceil(len(truth) * 0.05)))
    ranked = np.lexsort((np.arange(len(truth)), -prob))
    return float(np.mean(truth[ranked[:k]] > 1.0))


def sample_metrics(group: pd.DataFrame) -> dict:
    d = group['true_displacement'].to_numpy(float)
    pred = group['pred_displacement'].to_numpy(float)
    p = group['pred_perturbed_prob'].to_numpy(float)
    shell = group['shell_id'].to_numpy(int)
    radii = group['radii'].to_numpy(float)
    z = d > 1.0
    active = p * pred > 0.1
    pred_rho = float(radii[active].max()) if active.any() else 0.0
    truth_rho = float(radii[z].max()) if z.any() else 0.0
    positions = group.loc[z, 'residue_index'].to_numpy(int)
    runs = np.split(positions, np.where(np.diff(positions) != 1)[0] + 1) if len(positions) else []
    near = shell <= 1
    background = shell >= 2
    result = {
        'sample_id': str(group['sample_id'].iat[0]),
        'cluster_id_30': str(group['cluster_id_30'].iat[0]),
        'length': int(len(d)),
        'max_true_displacement': float(d.max()),
        'n_perturbed': int(z.sum()),
        'n_far_s2_s3': int((z & ((shell == 2) | (shell == 3))).sum()),
        'n_far_s4': int((z & (shell == 4)).sum()),
        'longest_perturbed_run': max((len(run) for run in runs), default=0),
        'true_radius': truth_rho,
        'true_class': 0 if not z.any() else (1 if truth_rho <= 8.0 else 2),
        'pred_radius': pred_rho,
        'pred_class': 0 if pred.max() <= 1.0 else (1 if pred_rho <= 8.0 else 2),
        'radius_error': abs(pred_rho - truth_rho),
        'auprc': auprc(d, p),
        'shell_mae': shell_mae(d, pred, shell),
        'near_mae': float(np.mean(np.abs(d[near] - pred[near]))),
        'near_max_displacement': float(d[near].max()),
        'background_mean': float(d[background].mean()),
        'background_sd': float(d[background].std()),
        'precision_at5': precision_at5(d, p),
    }
    return result


def scan_model(name: str) -> pd.DataFrame:
    frames = [sample_metrics(group) for _, group in sample_groups(MODEL_PATHS[name])]
    return pd.DataFrame(frames)


def selected_prediction_rows(path: Path, ids: set[str]) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, chunksize=250000, dtype={'sample_id': str, 'cluster_id_30': str}):
        keep = chunk[chunk['sample_id'].isin(ids)]
        if len(keep):
            parts.append(keep)
    return pd.concat(parts, ignore_index=True).sort_values(['sample_id', 'residue_index'])


def correlation(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    if np.std(truth) < 1e-12 or np.std(pred) < 1e-12:
        return float('nan'), float('nan')
    return float(pearsonr(truth, pred).statistic), float(spearmanr(truth, pred).statistic)


def ci95(values: np.ndarray, seed: int = 42) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    rng = np.random.default_rng(seed)
    sample_ids = rng.integers(0, len(a), size=(10000, len(a)))
    boot = a[sample_ids].mean(axis=1)
    return tuple(float(v) for v in np.quantile(boot, [0.025, 0.975]))
