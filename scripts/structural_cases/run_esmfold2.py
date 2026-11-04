"""Predict independent WT/Mut structures with Biohub ESMFold2."""
from __future__ import annotations

import pandas as pd
import torch
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    EsmFold2Model,
    ProteinInput,
    StructurePredictionInput,
)
from common import OUT


def main():
    pairs = pd.read_csv(OUT / 'fold_subset_221.csv', dtype={'sample_id': str})
    output = OUT / 'esmfold2_predictions'
    output.mkdir(parents=True, exist_ok=True)

    model = EsmFold2Model.from_pretrained('biohub/ESMFold2', device='cuda').eval()
    builder = ESMFold2InputBuilder()

    with torch.inference_mode():
        for index, pair in enumerate(pairs.itertuples(index=False), 1):
            for tag in ('wt', 'mut'):
                path = output / f'{pair.fold_key}_{tag}.cif'
                if path.exists():
                    continue
                fold_input = StructurePredictionInput(
                    sequences=[ProteinInput(id='A', sequence=getattr(pair, f'{tag}_sequence'))]
                )
                result = builder.fold(
                    model,
                    fold_input,
                    num_loops=20,
                    num_sampling_steps=100,
                    num_diffusion_samples=1,
                    seed=42,
                )
                path.write_text(result.complex.to_mmcif(), encoding='utf-8')
            print(f'ESMFold2: {index}/{len(pairs)} {pair.sample_id}', flush=True)


if __name__ == '__main__':
    main()
