#!/usr/bin/env bash
set -euo pipefail

TMALIGN_BIN="${TMALIGN_BIN:-}"

if [ -n "${TMALIGN_BIN}" ]; then
  python scripts/build_alignment_sensitivity_data.py \
    --base-config configs/c1000/base_v5.yaml \
    --all \
    --tmalign-bin "${TMALIGN_BIN}"
else
  echo "WARNING: TMALIGN_BIN is empty; building only the three Kabsch variants."
  python scripts/build_alignment_sensitivity_data.py \
    --base-config configs/c1000/base_v5.yaml \
    --all
fi
