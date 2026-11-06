"""Audit the extreme non-local example and choose two quality-controlled examples.

Run from the mutation_shell repository root, after select_cases.py.
Reads existing predictions, sample tensors, and deposited mmCIF chains.
No training or folding inference.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import OUT, MODEL_PATHS, sample_groups
from musrnet.alignment import kabsch_transform
from musrnet.pdb_io import inspect_chain

SAMPLES = Path('data/processed/samples')


def inspect_pair(sample_id: str, pdb_dir: Path) -> tuple[dict, pd.DataFrame]:
    s = torch.load(SAMPLES / f'{sample_id}.pt', map_location='cpu', weights_only=False)
    wt = s['coords_wt'].numpy().astype(float)
    mut = s['coords_mut'].numpy().astype(float)
    radii = s['radii'].numpy().astype(float)
    d = s['displacement'].numpy().astype(float)
    aligned, _, _, rmsd_4a = kabsch_transform(wt, mut, radii > 4.0)
    _, _, _, rmsd_all = kabsch_transform(wt, mut, np.ones(len(d), dtype=bool))
    delta = np.linalg.norm(aligned - mut, axis=1)
    assert np.max(np.abs(delta - d)) < 0.02, f'Inconsistent displacement labels: {sample_id}'
    assert len(s['wt_sequence']) == len(s['mut_sequence']) == len(d)
    assert sum(a != b for a, b in zip(s['wt_sequence'], s['mut_sequence'])) == 1
    assert s['wt_sequence'][int(s['mut_pos'])] != s['mut_sequence'][int(s['mut_pos'])]

    info = {'sample_id': sample_id, 'length': len(d), 'mutation_index_1based': int(s['mut_pos']) + 1,
            'wt_id': s['wt_pdb_id'], 'wt_chain': s['wt_chain_id'],
            'mut_id': s['mut_pdb_id'], 'mut_chain': s['mut_chain_id'],
            'mean_displacement': float(d.mean()), 'median_displacement': float(np.median(d)),
            'max_displacement': float(d.max()), 'p95_displacement': float(np.quantile(d, .95)),
            'n_res_gt_1': int((d > 1).sum()), 'n_res_gt_10': int((d > 10).sum()),
            'n_res_gt_50': int((d > 50).sum()),
            'rmsd_exclude_4a': rmsd_4a, 'rmsd_all_residues': rmsd_all,
            'distance_matrix_mae': float(np.mean(np.abs(pdist(wt) - pdist(mut))))}
    for tag, coords in (('wt', wt), ('mut', mut)):
        step = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        info[f'{tag}_max_ca_step'] = float(step.max())
        info[f'{tag}_n_ca_breaks_gt_4p6'] = int((step > 4.6).sum())
        pdb_id = str(s[f'{tag}_pdb_id']).lower()
        chain_id = s[f'{tag}_chain_id']
        cif = pdb_dir / f'{pdb_id}.cif.gz'
        chain_info = inspect_chain(cif, chain_id, 'mmcif', str(pdb_dir.parent.parent), pdb_dir.parent.name)
        info[f'{tag}_missing_ca_count'] = int(chain_info['missing_ca_count'])
        assert len(chain_info['coords']) == len(d), f'{tag} modeled residues differ: {sample_id}'
        assert chain_info['sequence'] == s[f'{tag}_sequence'], f'{tag} residue sequence differs: {sample_id}'
        with gzip.open(cif, 'rt') as fh:
            header = fh.read(125000)
        info[f'{tag}_xray'] = 'X-RAY DIFFRACTION' in header.upper()
        import re
        hit = re.search(r'_refine\.ls_d_res_high\s+([0-9.]+)', header)
        info[f'{tag}_resolution'] = float(hit.group(1)) if hit else np.nan

    shell_df = pd.DataFrame({'shell': s['shell_id'].numpy().astype(int), 'd': d})
    shells = shell_df.groupby('shell', as_index=False).agg(
        n_residues=('d', 'size'), mean_displacement=('d', 'mean'),
        median_displacement=('d', 'median'), max_displacement=('d', 'max'),
        n_gt_1=('d', lambda x: int((x > 1).sum())),
        n_gt_10=('d', lambda x: int((x > 10).sum())),
        n_gt_50=('d', lambda x: int((x > 50).sum())))
    return info, shells


def passes_quality(sample_id: str, pdb_dir: Path, max_resolution: float) -> tuple[bool, dict]:
    info, _ = inspect_pair(sample_id, pdb_dir)
    ok = all(info[f'{tag}_n_ca_breaks_gt_4p6'] == 0
             and info[f'{tag}_missing_ca_count'] == 0
             and info[f'{tag}_xray']
             and np.isfinite(info[f'{tag}_resolution'])
             and info[f'{tag}_resolution'] < max_resolution
             for tag in ('wt', 'mut'))
    return ok, info


def first_quality_case(candidates: pd.DataFrame, pdb_dir: Path, max_resolution: float) -> tuple[dict, pd.DataFrame]:
    reviews = []
    for r in candidates.itertuples(index=False):
        ok, info = passes_quality(str(r.sample_id), pdb_dir, max_resolution)
        reviews.append({**info, 'quality_pass': bool(ok)})
        if ok:
            return r._asdict(), pd.DataFrame(reviews)
    raise RuntimeError('No candidate passed experimental structure quality requirements')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pdb-dir', type=Path, required=True)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(OUT / 'selected_cases.csv', dtype={'sample_id': str, 'cluster_id_30': str})
    current = selected.set_index('case')
    old_nonlocal_id = str(current.loc['case3_nonlocal', 'sample_id'])
    selected_ids = set(selected.sample_id)
    excluded_clusters = set(selected.cluster_id_30.astype(str))
    musr = pd.read_csv(OUT / 'case_sample_metrics_musrnet.csv', dtype={'sample_id': str, 'cluster_id_30': str})
    geom = pd.read_csv(OUT / 'case_sample_metrics_geometry.csv', dtype={'sample_id': str})
    geom = geom[['sample_id', 'auprc', 'shell_mae']].rename(columns={
        'auprc': 'geometry_auprc', 'shell_mae': 'geometry_shell_mae'})
    candidates = musr.merge(geom, on='sample_id', validate='one_to_one')
    candidates['auprc_advantage'] = candidates.auprc - candidates.geometry_auprc
    candidates['mae_advantage'] = candidates.geometry_shell_mae - candidates.shell_mae

    # Sample-maximum displacement is quantified across the fixed full test set.
    maxvals = candidates.max_true_displacement.to_numpy(float)
    p95_max = float(np.quantile(maxvals, .95))
    cap = min(6.0, p95_max)
    distribution = {
        'test_pairs': int(len(candidates)), 'test_clusters': int(candidates.cluster_id_30.nunique()),
        'sample_max_p50': float(np.quantile(maxvals, .5)),
        'sample_max_p90': float(np.quantile(maxvals, .9)),
        'sample_max_p95': p95_max, 'sample_max_p99': float(np.quantile(maxvals, .99)),
        'nonlocal_display_max_cap': cap,
    }
    for threshold in (5., 10., 20., 50.):
        x = candidates.max_true_displacement >= threshold
        distribution[f'n_pairs_max_ge_{int(threshold)}'] = int(x.sum())
        distribution[f'pct_pairs_max_ge_{int(threshold)}'] = float(100 * x.mean())
        distribution[f'n_clusters_max_ge_{int(threshold)}'] = int(candidates.loc[x, 'cluster_id_30'].nunique())
    candidates.nlargest(100, 'max_true_displacement').to_csv(OUT / 'extreme_displacement_pairs.csv', index=False)
    (OUT / 'displacement_distribution.json').write_text(json.dumps(distribution, indent=2))

    old_info, old_shells = inspect_pair(old_nonlocal_id, args.pdb_dir)
    pd.DataFrame([old_info]).to_csv(OUT / 'original_nonlocal_structure_audit.csv', index=False)
    old_shells.to_csv(OUT / 'original_nonlocal_shells.csv', index=False)
    distribution.update({f'original_nonlocal_{k}': v for k, v in old_info.items()
                         if k in ('max_displacement', 'mean_displacement', 'n_res_gt_50')})

    # First filter on experimental response and model performance. No percentile/metric
    # is recalculated after choosing a case; the selection is illustrative only.
    nonlocal_base = candidates[
        (candidates.true_class == 2)
        & (candidates.true_radius > 12.0)
        & (candidates.n_far_s2_s3 >= 3)
        & (candidates.max_true_displacement >= 1.5)
        & (candidates.max_true_displacement <= cap)
        & (~candidates.cluster_id_30.isin(excluded_clusters))
    ].copy()
    nonlocal_candidates = nonlocal_base[
        (nonlocal_base.pred_class == 2)
        & (nonlocal_base.radius_error < 2.0)
        & (nonlocal_base.auprc >= 0.40)
        & (nonlocal_base.auprc_advantage >= 0.03)
        & (nonlocal_base.mae_advantage >= 0)
    ].sort_values(['auprc_advantage', 'auprc', 'shell_mae', 'radius_error'],
                  ascending=[False, False, True, True])
    nonlocal_candidates.to_csv(OUT / 'nonlocal_replacement_candidates.csv', index=False)

    # Supplemental example preserves strong localization while recovering its
    # observed displacement peak at the SAME residue.
    peak_base = candidates[
        candidates.length.between(120, 300)
        & candidates.max_true_displacement.between(2.0, 4.5)
        & candidates.n_perturbed.between(5, 20)
        & (candidates.auprc >= 0.65)
        & (candidates.auprc_advantage >= 0.10)
        & (~candidates.cluster_id_30.isin(excluded_clusters))
    ].copy()
    peak_ids = set(peak_base.sample_id)
    peak_rows = []
    residue_counts = {1: 0, 2: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    total_residues = 0
    for sid, g in sample_groups(MODEL_PATHS['musrnet']):
        d = g.true_displacement.to_numpy(float)
        n = len(d)
        total_residues += n
        for threshold in residue_counts:
            residue_counts[threshold] += int((d > threshold).sum())
        if sid not in peak_ids:
            continue
        pred = g.pred_displacement.to_numpy(float)
        i = int(np.argmax(d))
        j = int(np.argmax(pred))
        peak_rows.append({
            'sample_id': sid,
            'true_peak_index': i, 'pred_peak_index': j,
            'true_peak': float(d[i]), 'pred_at_true_peak': float(pred[i]),
            'pred_global_max': float(pred[j]),
            'peak_site_relative_error': float(abs(pred[i] - d[i]) / d[i]),
            'peak_magnitude_relative_error': float(abs(pred.max() - d.max()) / d.max()),
            'peak_index_distance': abs(i - j),
        })
    distribution['total_residues'] = total_residues
    for threshold, count in residue_counts.items():
        distribution[f'n_residues_gt_{threshold}'] = count
        distribution[f'pct_residues_gt_{threshold}'] = 100 * count / total_residues
    (OUT / 'displacement_distribution.json').write_text(json.dumps(distribution, indent=2))

    peak_candidates = peak_base.merge(pd.DataFrame(peak_rows), on='sample_id', validate='one_to_one')
    peak_candidates = peak_candidates[
        (peak_candidates.peak_site_relative_error <= 0.25)
        & (peak_candidates.peak_magnitude_relative_error <= 0.35)
        & (peak_candidates.peak_index_distance <= 5)
    ].sort_values(['peak_site_relative_error', 'auprc', 'shell_mae'],
                  ascending=[True, False, True])
    peak_candidates.to_csv(OUT / 'peak_recovery_candidates.csv', index=False)

    counts = pd.DataFrame([
        {'stage': 'nonlocal_response_and_moderate_amplitude', 'n_samples': len(nonlocal_base)},
        {'stage': 'nonlocal_and_correct_class_radius_model_better', 'n_samples': len(nonlocal_candidates)},
        {'stage': 'peak_example_response_and_auprc', 'n_samples': len(peak_base)},
        {'stage': 'peak_example_with_peak_recovery', 'n_samples': len(peak_candidates)},
    ])
    counts.to_csv(OUT / 'review_filter_counts.csv', index=False)
    print(pd.DataFrame([distribution]).to_string(index=False))
    print(counts.to_string(index=False))
    assert len(nonlocal_candidates), 'No correct non-local candidate passed declared thresholds; review candidates and report this.'
    assert len(peak_candidates), 'No peak-recovery candidate passed declared thresholds; review candidates and report this.'

    replacement, q_nonlocal = first_quality_case(nonlocal_candidates, args.pdb_dir, 3.0)
    peak, q_peak = first_quality_case(peak_candidates, args.pdb_dir, 3.0)
    q_nonlocal.to_csv(OUT / 'nonlocal_candidate_structure_checks.csv', index=False)
    q_peak.to_csv(OUT / 'peak_candidate_structure_checks.csv', index=False)
    assert str(replacement['cluster_id_30']) != str(peak['cluster_id_30'])
    replacement['case'] = 'case3_nonlocal'
    peak['case'] = 'case_peak_supplement'
    pd.DataFrame([replacement]).to_csv(OUT / 'selected_nonlocal_replacement.csv', index=False)
    pd.DataFrame([peak]).to_csv(OUT / 'selected_peak_supplement.csv', index=False)
    print('Non-local replacement:', replacement['sample_id'])
    print('Peak recovery supplement:', peak['sample_id'])
    print('Original non-local maximum C-alpha displacement:', old_info['max_displacement'])


if __name__ == '__main__':
    main()
