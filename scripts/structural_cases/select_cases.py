"""Strictly select four distinct-cluster cases from existing test predictions."""
from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path
import os
from multiprocessing import RLock
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import OUT, scan_model

ANNOT = Path('outputs/call/biological_stratification/v5_seed42_vs_geometry_gnn/biological_annotations.csv')
SAMPLE_DIR = Path('data/processed/samples')
PDB_DIR = Path('data/pdb')


def crystal_resolution(pdb_id: str) -> float:
    with gzip.open(PDB_DIR / f'{pdb_id.lower()}.cif.gz', 'rt') as fh:
        header = fh.read(100000)
    if not re.search(r'X-RAY DIFFRACTION', header, re.I):
        return float('nan')
    hit = re.search(r'_refine\.ls_d_res_high\s+([0-9.]+)', header)
    return float(hit.group(1)) if hit else float('nan')


def contiguous(sample_id: str) -> bool:
    sample = torch.load(SAMPLE_DIR / f'{sample_id}.pt', map_location='cpu', weights_only=False)
    for key in ('coords_wt', 'coords_mut'):
        ca = sample[key].numpy()
        if not np.all(np.linalg.norm(np.diff(ca, axis=0), axis=1) < 4.6):
            return False
    return True


def enrich_and_choose(label: str, candidates: pd.DataFrame, used_clusters: set[str],
                      verify_crystal: bool = False) -> dict:
    print(f'{label}: {len(candidates)} numeric candidates')
    OUT.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUT / f'candidates_{label}.csv', index=False)
    rejected = {'used_cluster': 0, 'chain_break': 0, 'crystal_resolution': 0}
    for row in tqdm(
        candidates.itertuples(index=False),
        total=len(candidates),
        desc=f'Selecting {label}',
        unit='candidate',
    ):
        if str(row.cluster_id_30) in used_clusters:
            rejected['used_cluster'] += 1
            continue
        if not contiguous(str(row.sample_id)):
            rejected['chain_break'] += 1
            continue
        rec = row._asdict()
        if verify_crystal:
            wt_res = crystal_resolution(str(row.wt_pdb_id))
            mut_res = crystal_resolution(str(row.mut_pdb_id))
            if not (np.isfinite(wt_res) and wt_res < 2.0 and np.isfinite(mut_res) and mut_res < 2.0):
                rejected['crystal_resolution'] += 1
                continue
            rec['wt_resolution'] = wt_res
            rec['mut_resolution'] = mut_res
        used_clusters.add(str(row.cluster_id_30))
        rec['case'] = label
        return rec
    raise RuntimeError(
        f'{label}: {len(candidates)} numeric candidates, none passed structural checks. '
        f'Rejections: {rejected}. See candidates_{label}.csv.'
    )


def main():
    print('Summarizing five existing test prediction CSVs; no model inference performed.')
    model_names = (
        'musrnet',
        'geometry',
        'coordinate',
        'global_loss',
        'direct',
    )

    results = {}

    progress_lock = RLock()
    tqdm.set_lock(progress_lock)

    with ProcessPoolExecutor(
        max_workers=min(3, os.cpu_count() or 1),
        initializer=tqdm.set_lock,
        initargs=(progress_lock,),
    ) as executor:
        futures = {
            executor.submit(
                scan_model,
                name,
                progress=True,
                position=i + 1,
            ): name
            for i, name in enumerate(model_names)
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc='Scanning models',
            unit='model',
            position=0,
        ):
            name = futures[future]
            results[name] = future.result()

    full = results['musrnet']
    geometry = results['geometry']
    coord = results['coordinate']
    glob = results['global_loss']
    direct = results['direct']
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in [('musrnet', full), ('geometry', geometry), ('coordinate', coord),
                     ('global_loss', glob), ('direct', direct)]:
        df.to_csv(OUT / f'case_sample_metrics_{name}.csv', index=False)
    annot = pd.read_csv(ANNOT, dtype={'sample_id': str, 'cluster_id_30': str}).drop_duplicates('sample_id')
    annot = annot[['sample_id', 'exposure_group', 'charge_group', 'resolution',
                   'wt_pdb_id', 'wt_chain_id', 'mut_pdb_id', 'mut_chain_id',
                   'wt_aa_type', 'mut_aa_type', 'rsa_mut_site', 'secondary_structure_group']]
    all_cases = full.merge(annot, on='sample_id', validate='one_to_one')
    for name, df in [('geometry', geometry), ('coordinate', coord), ('global', glob), ('direct', direct)]:
        sub = df[['sample_id', 'shell_mae', 'auprc', 'near_mae']].rename(columns={
            'shell_mae': f'shell_mae_{name}', 'auprc': f'auprc_{name}', 'near_mae': f'near_mae_{name}'})
        all_cases = all_cases.merge(sub, on='sample_id', validate='one_to_one')

    cases = []
    used = set()
    # Mutation-site exposure and charge-change subgroups were assessed separately
    # in the stratification analysis. Their intersection is preferred, not required.
    buried = all_cases.exposure_group.eq('buried')
    charge = all_cases.charge_group.eq('neutral_to_charged')
    joint = buried & charge
    either = buried | charge

    structure = (
        all_cases.resolution.lt(2.0)
        & all_cases.length.between(120, 300)
        & all_cases.max_true_displacement.between(2.0, 4.5)
        & all_cases.n_perturbed.between(5, 20)
        & all_cases.longest_perturbed_run.ge(3)
    )
    improvement = (
        all_cases.auprc.gt(all_cases.auprc_geometry)
        & all_cases.shell_mae.lt(all_cases.shell_mae_coordinate)
    )
    original_metrics = (
        all_cases.auprc.gt(0.65)
        & (all_cases.auprc - all_cases.auprc_geometry).gt(0.20)
        & all_cases.shell_mae.lt(0.35)
        & (all_cases.shell_mae_coordinate - all_cases.shell_mae).gt(0.20)
    )

    counts = pd.DataFrame([
        {'stage': 'all_annotated', 'count': len(all_cases)},
        {'stage': 'buried_and_neutral_to_charged', 'count': int(joint.sum())},
        {'stage': 'either_subgroup_and_structure', 'count': int((either & structure).sum())},
        {'stage': 'either_subgroup_structure_and_improvement', 'count': int((either & structure & improvement).sum())},
        {'stage': 'original_all_conditions', 'count': int((
            joint & structure & original_metrics
            & all_cases.secondary_structure_group.isin(['loop', 'helix'])
        ).sum())},
    ])
    counts.to_csv(OUT / 'case1_filter_counts.csv', index=False)
    print(counts.to_string(index=False))

    a = all_cases.loc[either & structure & improvement].copy()
    a['joint_subgroup'] = joint.loc[a.index].astype(int)
    a['case1_subgroup'] = np.select(
        [joint.loc[a.index], buried.loc[a.index]],
        ['buried+neutral_to_charged', 'buried'],
        default='neutral_to_charged',
    )
    a['primary_advantage'] = a.auprc - a.auprc_geometry
    a['coordinate_advantage'] = a.shell_mae_coordinate - a.shell_mae
    a = a.sort_values(
        ['joint_subgroup', 'primary_advantage', 'auprc', 'coordinate_advantage'],
        ascending=[False, False, False, False],
    )
    cases.append(enrich_and_choose('case1_comparison', a, used, verify_crystal=True))

    # Case 2: keep the experimental Local definition as an inclusion criterion.
    # Predicted class and radius accuracy determine ranking, not ground truth.
    b = all_cases.loc[
        (all_cases.true_class == 1) & (all_cases.true_radius <= 8.0)
    ].copy()
    b['class_correct'] = b.pred_class.eq(1)
    b['radius_within_2A'] = b.radius_error.lt(2.0)
    local_counts = pd.DataFrame([
        {'stage': 'true_local', 'count': len(b)},
        {'stage': 'predicted_local', 'count': int(b.class_correct.sum())},
        {'stage': 'radius_error_under_2A', 'count': int(b.radius_within_2A.sum())},
        {'stage': 'both_requirements', 'count': int((b.class_correct & b.radius_within_2A).sum())},
    ])
    local_counts.to_csv(OUT / 'case2_filter_counts.csv', index=False)
    print(local_counts.to_string(index=False))
    b = b.sort_values(
        ['class_correct', 'radius_error', 'auprc', 'near_mae'],
        ascending=[False, True, False, True],
    )
    cases.append(enrich_and_choose('case2_local', b, used))

    # Case 3: retain genuinely non-local experimental responses.
    # Prefer correct class/radius predictions without fabricating radius accuracy.
    c = all_cases.loc[
        (all_cases.true_class == 2) & (all_cases.true_radius > 12.0)
        & (all_cases.n_far_s2_s3 >= 3)
    ].copy()
    c['class_correct'] = c.pred_class.eq(2)
    c['radius_within_2A'] = c.radius_error.lt(2.0)
    nonlocal_counts = pd.DataFrame([
        {'stage': 'true_nonlocal_with_far_response', 'count': len(c)},
        {'stage': 'predicted_nonlocal', 'count': int(c.class_correct.sum())},
        {'stage': 'radius_error_under_2A', 'count': int(c.radius_within_2A.sum())},
        {'stage': 'both_requirements', 'count': int((c.class_correct & c.radius_within_2A).sum())},
    ])
    nonlocal_counts.to_csv(OUT / 'case3_filter_counts.csv', index=False)
    print(nonlocal_counts.to_string(index=False))
    c = c.sort_values(
        ['class_correct', 'radius_within_2A', 'radius_error', 'n_far_s2_s3', 'auprc'],
        ascending=[False, False, True, False, False],
    )
    cases.append(enrich_and_choose('case3_nonlocal', c, used))

    # Case 4: require an observed proximal rearrangement. Rank the available
    # examples by measured ablation improvements and background properties.
    case4_length = all_cases.length.between(150, 250)
    case4_near = all_cases.near_max_displacement.ge(2.0)
    d = all_cases.loc[case4_length & case4_near].copy()
    d['gain_global'] = d.near_mae_global - d.near_mae
    d['gain_direct'] = d.near_mae_direct - d.near_mae
    d['min_near_gain'] = np.minimum(d.gain_global, d.gain_direct)
    d['both_better'] = d.min_near_gain.gt(0.0)
    d['both_strong'] = d.min_near_gain.ge(0.10)
    d['target_background'] = d.background_mean.between(0.3, 0.6)
    d['stable_background'] = d.background_sd.le(0.15)
    d['mechanism_gain'] = (d.gain_global + d.gain_direct) / 2

    case4_counts = pd.DataFrame([
        {'stage': 'all_annotated', 'count': len(all_cases)},
        {'stage': 'length_150_to_250', 'count': int(case4_length.sum())},
        {'stage': 'plus_proximal_displacement_ge_2A', 'count': len(d)},
        {'stage': 'plus_background_mean_0.3_to_0.6',
         'count': int(d.target_background.sum())},
        {'stage': 'plus_background_sd_le_0.15',
         'count': int((d.target_background & d.stable_background).sum())},
        {'stage': 'both_proximal_gains_positive',
         'count': int(d.both_better.sum())},
        {'stage': 'both_proximal_gains_ge_0.10A',
         'count': int(d.both_strong.sum())},
        {'stage': 'original_all_conditions',
         'count': int((d.target_background & d.stable_background & d.both_strong).sum())},
    ])
    case4_counts.to_csv(OUT / 'case4_filter_counts.csv', index=False)
    print(case4_counts.to_string(index=False))

    d = d.sort_values(
        ['both_strong', 'both_better', 'target_background',
         'stable_background', 'min_near_gain', 'background_sd', 'sample_id'],
        ascending=[False, False, False, False, False, True, True],
    )
    cases.append(enrich_and_choose('case4_mechanism', d, used))

    pd.DataFrame(cases).to_csv(OUT / 'selected_cases.csv', index=False)
    print(pd.DataFrame(cases)[[
        'case', 'sample_id', 'cluster_id_30', 'true_class', 'pred_class',
        'true_radius', 'pred_radius', 'radius_error', 'auprc', 'shell_mae',
    ]].to_string(index=False))


if __name__ == '__main__':
    main()
