"""Run paired WT/Mut ESMFold inference on fold_subset_221.csv (GPU)."""
from __future__ import annotations

import pandas as pd
import torch
import esm
from common import OUT


def main():
    df = pd.read_csv(OUT / 'fold_subset_221.csv', dtype={'sample_id': str})
    output = OUT / 'esmfold_predictions'
    output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    model = esm.pretrained.esmfold_v1().eval().cuda()
    model.set_chunk_size(128)
    for i, row in enumerate(df.itertuples(index=False), 1):
        for tag in ('wt', 'mut'):
            path = output / f'{row.fold_key}_{tag}.pdb'
            if path.exists():
                continue
            seq = getattr(row, f'{tag}_sequence')
            pdb_text = model.infer_pdb(seq)
            path.write_text(pdb_text)
        print(f'ESMFold: {i}/{len(df)} {row.sample_id}', flush=True)


if __name__ == '__main__':
    main()
