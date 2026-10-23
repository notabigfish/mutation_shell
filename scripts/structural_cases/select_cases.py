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
    candidates.head(1000).to_csv(OUT / f'candidates_{label}.csv', index=False)
    for row in tqdm(
        candidates.itertuples(index=False),
        total=len(candidates),
        desc=f'Selecting {label}',
        unit='candidate',
    ):
        if str(row.cluster_id_30) in used_clusters:
            continue
        if not contiguous(str(row.sample_id)):
            continue
        rec = row._asdict()
        if verify_crystal:
            wt_res = crystal_resolution(str(row.wt_pdb_id))
            mut_res = crystal_resolution(str(row.mut_pdb_id))
            if not (np.isfinite(wt_res) and wt_res < 2.0 and np.isfinite(mut_res) and mut_res < 2.0):
                continue
            rec['wt_resolution'] = wt_res
            rec['mut_resolution'] = mut_res
        used_clusters.add(str(row.cluster_id_30))
        rec['case'] = label
        return rec
    raise RuntimeError(f'No physically continuous {label} case passed all conditions; inspect candidates_{label}.csv')


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
    a = all_cases[
        (all_cases.exposure_group == 'buried') & (all_cases.charge_group == 'neutral_to_charged')
        & (all_cases.resolution < 2.0) & all_cases.length.between(120, 300)
        & all_cases.max_true_displacement.between(2.0, 4.5)
        & all_cases.n_perturbed.between(5, 20)
        & (all_cases.longest_perturbed_run >= 3)
        & (all_cases.secondary_structure_group.isin(['loop', 'helix']))
        & (all_cases.auprc > 0.65) & ((all_cases.auprc - all_cases.auprc_geometry) > 0.20)
        & (all_cases.shell_mae < 0.35) & ((all_cases.shell_mae_coordinate - all_cases.shell_mae) > 0.20)
    ].copy()
    a['primary_advantage'] = a.auprc - a.auprc_geometry
    a = a.sort_values(['primary_advantage', 'shell_mae'], ascending=[False, True])
    cases.append(enrich_and_choose('case1_comparison', a, used, verify_crystal=True))

    b = all_cases[
        (all_cases.true_class == 1) & (all_cases.true_radius <= 8.0)
        & (all_cases.pred_class == 1) & (all_cases.radius_error < 2.0)
    ].sort_values(['radius_error', 'auprc'], ascending=[True, False])
    cases.append(enrich_and_choose('case2_local', b, used))

    c = all_cases[
        (all_cases.true_class == 2) & (all_cases.true_radius > 12.0)
        & (all_cases.n_far_s2_s3 >= 3) & (all_cases.pred_class == 2)
        & (all_cases.radius_error < 2.0)
    ].sort_values(['n_far_s4', 'radius_error', 'auprc'], ascending=[False, True, False])
    cases.append(enrich_and_choose('case3_nonlocal', c, used))

    d = all_cases[
        all_cases.length.between(150, 250)
        & all_cases.background_mean.between(0.3, 0.6)
        & (all_cases.background_sd <= 0.15)
        & (all_cases.near_max_displacement >= 2.0)
        & ((all_cases.near_mae_global - all_cases.near_mae) >= 0.10)
        & ((all_cases.near_mae_direct - all_cases.near_mae) >= 0.10)
    ].copy()
    d['mechanism_gain'] = (d.near_mae_global + d.near_mae_direct) / 2 - d.near_mae
    d = d.sort_values('mechanism_gain', ascending=False)
    cases.append(enrich_and_choose('case4_mechanism', d, used))

    pd.DataFrame(cases).to_csv(OUT / 'selected_cases.csv', index=False)
    print(pd.DataFrame(cases)[['case', 'sample_id', 'cluster_id_30', 'length', 'auprc', 'shell_mae']].to_string(index=False))


if __name__ == '__main__':
    main()
