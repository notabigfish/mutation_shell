from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.dataset import get_sample_path, load_samples_manifest
from musrnet.train_utils import save_json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare residue labels between two alignment-specific manifests")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for loading samples")
    return parser.parse_args()



def load_samples(manifest_path: Path, num_workers: int) -> dict[str, dict]:
    manifest = load_samples_manifest(manifest_path)
    def _load(sample_id):
        return sample_id, torch.load(get_sample_path(manifest, sample_id), map_location="cpu")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return dict(
            tqdm(
                executor.map(_load, manifest["sample_ids"]), 
                total=len(manifest["sample_ids"]), 
                desc="Loading Samples"
            )
        )

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)


def main() -> None:
    args = parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = load_samples(PROJECT_ROOT / args.reference, args.num_workers)
    candidate = load_samples(PROJECT_ROOT / args.candidate, args.num_workers)
    common_ids = sorted(set(reference) & set(candidate))
    if not common_ids:
        raise ValueError("No common samples between reference and candidate manifests")

    all_ref_disp = []
    all_cand_disp = []
    all_ref_pert = []
    all_cand_pert = []
    radius_diffs = []
    class_disagreements = []
    shell_diff: dict[int, list[float]] = {k: [] for k in range(5)}
    per_sample_rows = []

    for sample_id in tqdm(common_ids):
        ref = reference[sample_id]
        cand = candidate[sample_id]
        ref_disp = ref["displacement"].cpu().numpy()
        cand_disp = cand["displacement"].cpu().numpy()
        if ref_disp.shape != cand_disp.shape:
            raise ValueError(f"Residue count mismatch for sample {sample_id}")
        ref_pert = ref["perturbed"].cpu().numpy().astype(bool)
        cand_pert = cand["perturbed"].cpu().numpy().astype(bool)
        ref_shell = ref["shell_id"].cpu().numpy()
        abs_diff = np.abs(ref_disp - cand_disp)

        all_ref_disp.append(ref_disp)
        all_cand_disp.append(cand_disp)
        all_ref_pert.append(ref_pert.astype(np.int64))
        all_cand_pert.append(cand_pert.astype(np.int64))
        radius_diffs.append(abs(float(ref["radius_label"][0]) - float(cand["radius_label"][0])))
        class_disagreements.append(float(int(ref["class_label"][0]) != int(cand["class_label"][0])))

        for shell_idx in range(5):
            mask = ref_shell == shell_idx
            if mask.any():
                shell_diff[shell_idx].extend(abs_diff[mask].tolist())

        union = np.logical_or(ref_pert, cand_pert).sum()
        jaccard = float(np.logical_and(ref_pert, cand_pert).sum() / union) if union else 1.0
        per_sample_rows.append(
            {
                "sample_id": sample_id,
                "cluster_id_30": str(ref["cluster_id_30"]),
                "mean_abs_diff_disp": float(abs_diff.mean()),
                "radius_abs_diff": radius_diffs[-1],
                "class_disagreement": class_disagreements[-1],
                "perturbed_jaccard": jaccard,
                "perturbed_flip_fraction": float(np.mean(ref_pert != cand_pert)),
            }
        )

    ref_disp_all = np.concatenate(all_ref_disp)
    cand_disp_all = np.concatenate(all_cand_disp)
    ref_pert_all = np.concatenate(all_ref_pert)
    cand_pert_all = np.concatenate(all_cand_pert)
    disp_abs_diff = np.abs(ref_disp_all - cand_disp_all)
    union = np.logical_or(ref_pert_all.astype(bool), cand_pert_all.astype(bool)).sum()
    summary = {
        "n_common_samples": len(common_ids),
        "mean_abs_diff_disp": float(disp_abs_diff.mean()),
        "median_abs_diff_disp": float(np.median(disp_abs_diff)),
        "pearson_corr_disp": safe_corr(ref_disp_all, cand_disp_all),
        "spearman_corr_disp": safe_corr(rankdata(ref_disp_all), rankdata(cand_disp_all)),
        "perturbed_jaccard": float(np.logical_and(ref_pert_all.astype(bool), cand_pert_all.astype(bool)).sum() / union) if union else 1.0,
        "perturbed_flip_fraction": float(np.mean(ref_pert_all != cand_pert_all)),
        "shell0_mean_abs_diff_disp": float(np.mean(shell_diff[0])) if shell_diff[0] else float("nan"),
        "shell1_mean_abs_diff_disp": float(np.mean(shell_diff[1])) if shell_diff[1] else float("nan"),
        "shell4_mean_abs_diff_disp": float(np.mean(shell_diff[4])) if shell_diff[4] else float("nan"),
        "radius_mean_abs_diff": float(np.mean(radius_diffs)),
        "class_disagreement_fraction": float(np.mean(class_disagreements)),
    }
    pd.DataFrame(per_sample_rows).to_csv(out_dir / "per_sample_label_diff.csv", index=False)
    save_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
