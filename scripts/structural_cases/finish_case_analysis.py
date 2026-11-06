"""Quantify far-field localization and select one mutation-proximal peak example."""
from pathlib import Path
import sys
import argparse
import gzip
import re
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import OUT, MODEL_PATHS, selected_prediction_rows
from musrnet.pdb_io import inspect_chain


def far_field():
    path = OUT / 'cases/case3_nonlocal'
    df = pd.read_csv(path / 'residues.csv')
    truth = df['true'].to_numpy(float) > 1.0
    pred = (df['prob_musrnet'].to_numpy(float) * df['musrnet'].to_numpy(float)) > 0.1
    shell = df['shell_id'].to_numpy(int)
    rows = []
    for label, mask in [(f'S{i}', shell == i) for i in (2, 3, 4)] + [('S2-S4', shell >= 2)]:
        tp = int(np.sum(mask & truth & pred))
        fp = int(np.sum(mask & ~truth & pred))
        fn = int(np.sum(mask & truth & ~pred))
        rows.append({'region': label, 'n_residues': int(mask.sum()), 'TP': tp, 'FP': fp, 'FN': fn,
                     'precision': tp / (tp + fp) if tp + fp else np.nan,
                     'recall': tp / (tp + fn) if tp + fn else np.nan,
                     'jaccard': tp / (tp + fp + fn) if tp + fp + fn else np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'nonlocal_far_localization.csv', index=False)
    print('\nNon-local localization: truth d>1 Å; predicted p*d>0.1 Å')
    print(result.to_string(index=False))


def crystal_resolution(pdb_dir, pdb_id):
    with gzip.open(pdb_dir / f'{str(pdb_id).lower()}.cif.gz', 'rt') as handle:
        head = handle.read(125000)
    if 'X-RAY DIFFRACTION' not in head.upper():
        return np.nan
    m = re.search(r'_refine\.ls_d_res_high\s+([0-9.]+)', head)
    return float(m.group(1)) if m else np.nan


def crystal_quality(sample_id, pdb_dir):
    s = torch.load(Path('data/processed/samples') / f'{sample_id}.pt', map_location='cpu', weights_only=False)
    for tag in ('wt', 'mut'):
        coords = s[f'coords_{tag}'].numpy()
        if np.any(np.linalg.norm(np.diff(coords, axis=0), axis=1) > 4.6):
            return False
        pdb_id = str(s[f'{tag}_pdb_id']).lower()
        if not (crystal_resolution(pdb_dir, pdb_id) < 3):
            return False
        chain = inspect_chain(pdb_dir / f'{pdb_id}.cif.gz', s[f'{tag}_chain_id'],
                              'mmcif', str(pdb_dir.parent.parent), pdb_dir.parent.name)
        if chain['missing_ca_count'] != 0 or chain['sequence'] != s[f'{tag}_sequence']:
            return False
    return True


def select_peak(pdb_dir):
    musr = pd.read_csv(OUT / 'case_sample_metrics_musrnet.csv',
                       dtype={'sample_id': str, 'cluster_id_30': str})
    geom = pd.read_csv(OUT / 'case_sample_metrics_geometry.csv', dtype={'sample_id': str})
    geom = geom[['sample_id', 'auprc']].rename(columns={'auprc': 'geometry_auprc'})
    selected = pd.read_csv(OUT / 'selected_cases.csv', dtype={'sample_id': str, 'cluster_id_30': str})
    cases = musr.merge(geom, on='sample_id', validate='one_to_one')
    candidates = cases.loc[
        cases.length.between(120, 300)
        & cases.max_true_displacement.between(2.0, 4.5)
        & cases.n_perturbed.between(5, 20)
        & cases.auprc.ge(0.65)
        & (cases.auprc - cases.geometry_auprc).ge(0.10)
        & ~cases.cluster_id_30.isin(selected.cluster_id_30)
    ].copy()
    print('Candidates before peak screening:', len(candidates))
    if candidates.empty:
        raise RuntimeError('No samples pass the fixed length, displacement and AUPRC filters.')
    ids = set(candidates.sample_id)
    prediction = selected_prediction_rows(MODEL_PATHS['musrnet'], ids)
    measured = []
    for sid, df in prediction.groupby('sample_id', sort=False):
        df = df.sort_values('residue_index')
        truth = df.true_displacement.to_numpy(float)
        pred = df.pred_displacement.to_numpy(float)
        i, j = int(truth.argmax()), int(pred.argmax())
        measured.append({'sample_id': sid, 'true_peak_radius_A': float(df.radii.iloc[i]),
                         'true_peak_pos': int(df.residue_index.iloc[i]) + 1,
                         'pred_peak_pos': int(df.residue_index.iloc[j]) + 1,
                         'peak_position_gap': abs(i-j),
                         'true_peak_A': float(truth[i]), 'pred_at_true_peak_A': float(pred[i]),
                         'local_peak_relative_error': float(abs(pred[i] - truth[i]) / truth[i]),
                         'global_peak_relative_error': float(abs(pred.max() - truth.max()) / truth.max())})
    screened = candidates.merge(pd.DataFrame(measured), on='sample_id', validate='one_to_one')
    screened = screened.sort_values(['local_peak_relative_error', 'auprc', 'shell_mae'],
                                    ascending=[True, False, True])
    screened.to_csv(OUT / 'peak_screening_all_candidates.csv', index=False)
    qualified = screened.loc[
        screened.true_peak_radius_A.le(8.0)
        & screened.peak_position_gap.le(5)
        & screened.local_peak_relative_error.le(0.25)
        & screened.global_peak_relative_error.le(0.35)
    ].copy()
    qualified.to_csv(OUT / 'peak_screening_eligible.csv', index=False)
    print('Eligible mutation-neighborhood peak cases:', len(qualified))
    for row in qualified.itertuples(index=False):
        if crystal_quality(str(row.sample_id), pdb_dir):
            output = qualified.loc[qualified.sample_id == str(row.sample_id)].head(1).copy()
            output.insert(0, 'case', 'case_peak_supplement')
            output.to_csv(OUT / 'selected_peak_supplement.csv', index=False)
            print('Selected case:', str(row.sample_id))
            return
    raise RuntimeError('No eligible peak case passed both WT and mutant crystal-quality checks. See peak_screening_eligible.csv.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb-dir', type=Path, required=True)
    args = parser.parse_args()
    far_field()
    select_peak(args.pdb_dir)


if __name__ == '__main__':
    main()
