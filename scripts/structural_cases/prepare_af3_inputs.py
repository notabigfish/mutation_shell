"""Write 442 single-chain AlphaFold3 jobs from the completed 221-pair subset."""
from __future__ import annotations

import json

import pandas as pd
from common import OUT


def main():
    pairs = pd.read_csv(OUT / 'fold_subset_221.csv', dtype={'sample_id': str})
    folder = OUT / 'af3_inputs'
    folder.mkdir(parents=True, exist_ok=True)
    (OUT / 'af3_predictions').mkdir(parents=True, exist_ok=True)

    jobs = []
    for pair in pairs.itertuples(index=False):
        for tag in ('wt', 'mut'):
            name = f'{pair.fold_key}_{tag}'
            payload = {
                'name': name,
                'sequences': [{'protein': {'id': 'A', 'sequence': getattr(pair, f'{tag}_sequence')}}],
                'modelSeeds': [42],
                'dialect': 'alphafold3',
                'version': 1,
            }
            path = folder / f'{name}.json'
            path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
            jobs.append(str(path))
    (OUT / 'af3_jobs.txt').write_text('\n'.join(jobs) + '\n', encoding='utf-8')
    print(f'Prepared {len(jobs)} single-chain AlphaFold3 jobs')


if __name__ == '__main__':
    main()
