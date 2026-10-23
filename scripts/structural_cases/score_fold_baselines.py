"""Score paired WT/Mut ESMFold and AF2 differential conformations and internal models."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from musrnet.alignment import kabsch_align
from common import OUT, MODEL_PATHS, selected_prediction_rows, shell_mae, auprc, correlation, ci95


def ca_from_pdb(path: Path) -> np.ndarray:
    structure = PDBParser(QUIET=True).get_structure('pred', str(path))
    residues = [r for r in next(structure.get_models()).get_residues() if 'CA' in r]
    return np.asarray([r['CA'].coord for r in residues], dtype=np.float64)


def af2_file(key: str, tag: str) -> Path:
    matches = sorted((OUT / 'af2_predictions').glob(f'{key}_{tag}*rank_001*.pdb'))
    assert len(matches) == 1, f'{key}_{tag}: expected one rank_001 AF2 PDB, got {len(matches)}'
    return matches[0]


def row_metrics(sid, cluster, model, d, pred, shell, score):
    pear, spear = correlation(d, pred)
    return {'sample_id': sid, 'cluster_id_30': cluster, 'model': model,
            'shell_mae': shell_mae(d, pred, shell), 'auprc': auprc(d, score),
            'pearson': pear, 'spearman': spear,
            'max_true_disp': d.max(), 'max_pred_disp': pred.max()}


def main():
    subset = pd.read_csv(OUT / 'fold_subset_221.csv', dtype={'sample_id': str, 'cluster_id_30': str})
    ids = set(subset.sample_id)
    predictions = {key: selected_prediction_rows(MODEL_PATHS[key], ids).set_index(['sample_id', 'residue_index'])
                   for key in ('musrnet', 'geometry', 'coordinate')}
    rows, residue_rows = [], []
    for item in subset.itertuples(index=False):
        sample = torch.load(f'data/processed/samples/{item.sample_id}.pt', map_location='cpu', weights_only=False)
        d = sample['displacement'].numpy().astype(float)
        shell = sample['shell_id'].numpy().astype(int)
        L = len(d)
        for model in ('musrnet', 'geometry', 'coordinate'):
            frame = predictions[model].loc[item.sample_id].sort_index()
            assert len(frame) == L
            pred = frame.pred_displacement.to_numpy(float)
            score = frame.pred_perturbed_prob.to_numpy(float)
            rows.append(row_metrics(item.sample_id, item.cluster_id_30, model, d, pred, shell, score))
        for model in ('esmfold', 'af2'):
            if model == 'esmfold':
                pred_wt = ca_from_pdb(OUT / 'esmfold_predictions' / f'{item.fold_key}_wt.pdb')
                pred_mut = ca_from_pdb(OUT / 'esmfold_predictions' / f'{item.fold_key}_mut.pdb')
            else:
                pred_wt = ca_from_pdb(af2_file(item.fold_key, 'wt'))
                pred_mut = ca_from_pdb(af2_file(item.fold_key, 'mut'))
            assert len(pred_wt) == len(pred_mut) == L, item.sample_id
            r = np.linalg.norm(pred_wt - pred_wt[item.mut_pos], axis=1)
            aligned, _, _ = kabsch_align(pred_wt, pred_mut, r > 4.0)
            pred = np.linalg.norm(aligned - pred_mut, axis=1)
            rows.append(row_metrics(item.sample_id, item.cluster_id_30, model, d, pred, shell, pred))
            residue_rows.extend({'sample_id': item.sample_id, 'residue_index': j, 'model': model,
                                 'pred_displacement': val} for j, val in enumerate(pred))
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / 'fold_baseline_per_sample.csv', index=False)
    pd.DataFrame(residue_rows).to_csv(OUT / 'fold_residue_fields.csv', index=False)
    summary = []
    for model, group in detail.groupby('model', sort=False):
        for metric in ('shell_mae', 'auprc', 'pearson', 'spearman'):
            vals = group[metric].dropna().to_numpy(float)
            low, high = ci95(vals)
            summary.append({'model': model, 'metric': metric, 'n_clusters': len(vals),
                            'cluster_mean': vals.mean(), 'ci95_low': low, 'ci95_high': high})
    pd.DataFrame(summary).to_csv(OUT / 'fold_baseline_summary.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
