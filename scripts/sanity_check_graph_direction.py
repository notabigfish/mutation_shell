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


def check_one_graph(data, k: int) -> None:
    pos = data.pos.float()
    edge_index = data.edge_index.long()
    src, dst = edge_index

    n = pos.size(0)
    if n <= 1:
        raise ValueError("Graph has <= 1 node")

    expected_k = min(k, n - 1)

    dist = torch.cdist(pos, pos)
    dist.fill_diagonal_(float("inf"))

    knn = dist.topk(expected_k, largest=False, dim=1).indices
    knn_sets = [set(knn[i].tolist()) for i in range(n)]

    incoming = [[] for _ in range(n)]
    for s, d in zip(src.tolist(), dst.tolist()):
        incoming[d].append(s)

    bad_nodes = []
    for center in range(n):
        got = set(incoming[center])
        expected = knn_sets[center]

        if got != expected:
            bad_nodes.append(
                {
                    "center": center,
                    "expected_first10": sorted(list(expected))[:10],
                    "got_first10": sorted(list(got))[:10],
                    "expected_count": len(expected),
                    "got_count": len(got),
                }
            )

    if bad_nodes:
        print("FAILED: edge_index direction or kNN construction is wrong.")
        print("Expected: src = neighbor, dst = center.")
        print("Example bad nodes:")
        for item in bad_nodes[:5]:
            print(item)
        raise SystemExit(1)

    print(
        f"PASSED: sample_id={data.sample_id}, nodes={n}, "
        f"edges={edge_index.size(1)}, expected_in_degree={expected_k}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()

    config = load_yaml(args.config)
    samples_manifest = load_samples_manifest(config["paths"]["samples"])

    sample_ids = [m["sample_id"] for m in samples_manifest["metadata"][: args.num_samples]]
    dataset = MuSRNetDataset(samples_manifest, sample_ids, int(config["data"]["knn_k"]))

    for i in range(len(dataset)):
        data = dataset[i]
        check_one_graph(data, k=int(config["data"]["knn_k"]))

    print("ALL PASSED.")


if __name__ == "__main__":
    main()