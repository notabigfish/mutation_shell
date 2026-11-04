"""Reproduce the counterfactual one-per-cluster rule, insert four chosen cases."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from musrnet.dataset import load_samples_manifest
from common import OUT


def main():
    selected = pd.read_csv(OUT / 'selected_cases.csv', dtype={'sample_id': str, 'cluster_id_30': str})
    manifest = load_samples_manifest('data/processed/samples.pt')
    with open('data/processed/splits_call.json', encoding='utf-8') as f:
        test_ids = json.load(f)['test']
    cluster_by_id = {str(meta['sample_id']): str(meta['cluster_id_30']) for meta in manifest['metadata']}

    first = {}
    for sid in test_ids:  # identical selection order to scripts/counterfactual_tests.py
        if cluster_by_id[str(sid)] not in first:
            first[cluster_by_id[str(sid)]] = str(sid)
    for r in selected.itertuples(index=False):
        assert cluster_by_id[str(r.sample_id)] == str(r.cluster_id_30)
        first[str(r.cluster_id_30)] = str(r.sample_id)
    assert len(first) == 221, f'Expected 221 test clusters, got {len(first)}'
    rows = []
    for i, (cluster, sid) in enumerate(first.items(), start=1):
        s = torch.load(Path(manifest['samples_dir']) / f'{sid}.pt', map_location='cpu', weights_only=False)
        key = f'pair{i:04d}'
        rows.append({'fold_key': key, 'sample_id': sid, 'cluster_id_30': cluster,
                     'wt_sequence': s['wt_sequence'], 'mut_sequence': s['mut_sequence'],
                     'mut_pos': int(s['mut_pos']), 'length': len(s['wt_sequence'])})
    pd.DataFrame(rows).to_csv(OUT / 'fold_subset_221.csv', index=False)
    print(f'Wrote {len(rows)} WT/Mut pairs')


if __name__ == '__main__':
    main()
