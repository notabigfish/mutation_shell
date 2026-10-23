#!/usr/bin/env bash
set -euo pipefail
# ColabFold uses AlphaFold2-ptm without template mode; one model and three recycles.
colabfold_batch \
  --model-type alphafold2_ptm \
  --num-models 1 \
  --num-recycle 3 \
  --random-seed 42 \
  results/structural_cases/af2_fasta \
  results/structural_cases/af2_predictions
