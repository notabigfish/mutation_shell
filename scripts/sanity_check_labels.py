from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.dataset import MuSRNetDataset, load_samples_manifest
from musrnet.train_utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    config = load_yaml(args.config)
    samples_manifest = load_samples_manifest(config["paths"]["samples"])

    sample_ids = [m["sample_id"] for m in samples_manifest["metadata"][: args.num_samples]]
    dataset = MuSRNetDataset(samples_manifest, sample_ids, int(config["data"]["knn_k"]))

    for idx in range(len(dataset)):
        data = dataset[idx]

        n = data.x_basic.size(0)

        checks = {
            "pos": data.pos.size(0),
            "y_disp": data.y_disp.size(0),
            "y_perturbed": data.y_perturbed.size(0),
            "shell_id": data.shell_id.size(0),
            "esm_wt": data.esm_wt.size(0),
            "esm_delta": data.esm_delta.size(0),
        }

        for name, length in checks.items():
            if length != n:
                raise ValueError(
                    f"Length mismatch sample={data.sample_id}: "
                    f"x_basic={n}, {name}={length}"
                )

        if torch.isnan(data.y_disp).any():
            raise ValueError(f"NaN in y_disp: {data.sample_id}")

        if torch.isnan(data.esm_wt).any():
            raise ValueError(f"NaN in esm_wt: {data.sample_id}")

        if not torch.all((data.shell_id >= 0) & (data.shell_id <= 4)):
            raise ValueError(f"Invalid shell_id: {data.sample_id}")

        if not torch.all((data.y_perturbed == 0) | (data.y_perturbed == 1)):
            raise ValueError(f"Invalid y_perturbed: {data.sample_id}")

    print("PASSED: labels/features have consistent node lengths.")


if __name__ == "__main__":
    main()