from __future__ import annotations
"""
python scripts/prepare_data.py --csv data/SingleMutPairs2024_subset_c50.csv  --out data/processed/samples_subset_c50_raw.pt
"""
import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import os
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.labels import build_structural_labels
from musrnet.pdb_io import find_pdb_file, inspect_chain

from concurrent.futures import ProcessPoolExecutor, as_completed

def build_sample(row: dict[str, Any], pdb_format: str, pdb_root: str, pdb_version: str) -> tuple[dict[str, Any] | None, str | None]:
	try:
		wt_pdb_path = find_pdb_file(os.path.join(pdb_root, pdb_version, 'pdb'), str(row["wt_pdb_id"]).lower())
		mut_pdb_path = find_pdb_file(os.path.join(pdb_root, pdb_version, 'pdb'), str(row["mut_pdb_id"]).lower())
	except FileNotFoundError:
		return row['sample_id'], "missing PDB file"

	try:
		wt_info = inspect_chain(wt_pdb_path, str(row["wt_chain_id"]), pdb_format, pdb_root, pdb_version)
		mut_info = inspect_chain(mut_pdb_path, str(row["mut_chain_id"]), pdb_format, pdb_root, pdb_version)
	except KeyError:
		return row['sample_id'], "missing chain"

	if wt_info["missing_ca_count"] > 0 or mut_info["missing_ca_count"] > 0:
		return row['sample_id'], "missing C-alpha"

	wt_sequence = str(row["wt_sequence"])
	mut_sequence = str(row["mut_sequence"])
	if wt_info["sequence"] != wt_sequence or mut_info["sequence"] != mut_sequence:
		return row['sample_id'], "sequence mismatch"
	if len(wt_sequence) != len(mut_sequence):
		return row['sample_id'], "sequence mismatch"

	diff_positions = [idx for idx, (a, b) in enumerate(zip(wt_sequence, mut_sequence)) if a != b]
	if len(diff_positions) != 1:
		return row['sample_id'], "not single mutation"

	mut_pos = int(row["mut_pos_seq_index"])
	if diff_positions[0] != mut_pos or not (0 <= mut_pos < len(wt_sequence)):
		return row['sample_id'], "invalid mutation position index"

	wt_aa = str(row["wt_aa_type"])
	mut_aa = str(row["mut_aa_type"])
	if wt_sequence[mut_pos] != wt_aa or mut_sequence[mut_pos] != mut_aa:
		return row['sample_id'], "invalid mutation position residue"
	# if wt_info["residues"][mut_pos]["pdb_number"] != int(row["mut_pos_pdb_number"]):
	# 	return row['sample_id'], "invalid mutation position PDB number"

	coords_wt = wt_info["coords"]
	coords_mut = mut_info["coords"]
	if coords_wt.shape != coords_mut.shape:
		return row['sample_id'], "sequence mismatch"

	try:
		labels = build_structural_labels(coords_wt=coords_wt, coords_mut=coords_mut, mut_pos=mut_pos)
	except ValueError:
		return row['sample_id'], "failed alignment"

	sample = {
		"sample_id": str(row["sample_id"]),
		"wt_pdb_id": str(row["wt_pdb_id"]).lower(),
		"wt_chain_id": str(row["wt_chain_id"]),
		"mut_pdb_id": str(row["mut_pdb_id"]).lower(),
		"mut_chain_id": str(row["mut_chain_id"]),
		"mut_pos": mut_pos,
		"wt_aa": wt_aa,
		"mut_aa": mut_aa,
		"wt_sequence": wt_sequence,
		"mut_sequence": mut_sequence,
		"cluster_id_30": str(row["cluster_id_30"]),
		"release_date": str(row["release_date"]),
		"coords_wt": torch.from_numpy(coords_wt).float(),
		"coords_mut": torch.from_numpy(coords_mut).float(),
		"coords_wt_aligned": torch.from_numpy(labels["coords_wt_aligned"]).float(),
		"displacement": torch.from_numpy(labels["displacement"]).float(),
		"radii": torch.from_numpy(labels["radii"]).float(),
		"shell_id": torch.from_numpy(labels["shell_id"]).long(),
		"perturbed": torch.from_numpy(labels["perturbed"]).float(),
		"radius_label": torch.from_numpy(labels["radius_label"]).float(),
		"class_label": torch.from_numpy(labels["class_label"]).long(),
	}
	return sample, None

def process_row_worker(row: dict, pdb_format: str, pdb_root: Path, pdb_version: str, sample_dir: Path, resume: bool=False) -> dict:
	sample_id = str(row["sample_id"])
	sample_path = sample_dir / f"{sample_id}.pt"
	if resume and sample_path.exists():
		try:
			sample = torch.load(sample_path, map_location="cpu", weights_only=False)
			meta = {
				"sample_id": sample["sample_id"],
				"cluster_id_30": sample["cluster_id_30"],
				"release_date": sample["release_date"],
				"length": int(sample["coords_wt"].shape[0]),
			}
			return {"status": "success", "sample_id": sample["sample_id"], "metadata": meta}
		except Exception:
			pass

	sample, reason = build_sample(row, pdb_format, pdb_root, pdb_version)
	if isinstance(sample, str):
		return {"status": "rejected", "sample_id": sample, "reason": f'{sample} rejected: {reason}'}
	temp_path = sample_path.with_suffix(".pt.tmp")
	torch.save(sample, temp_path)
	temp_path.rename(sample_path)

	meta = {
		"sample_id": sample["sample_id"],
		"cluster_id_30": sample["cluster_id_30"],
		"release_date": sample["release_date"],
		"length": int(sample["coords_wt"].shape[0]),
	}
	return {"status": "success", "sample_id": sample["sample_id"], "metadata": meta}

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Prepare MuSRNet structural samples")
	parser.add_argument('--pdb_root', type=str, default='/rds/projects/l/liuje-multiai/shuo/datasets', help='Directory containing PDB files')
	parser.add_argument('--pdb_version', type=str, default='pdb_260603', help='pdb_241028 pdb_260603')
	parser.add_argument('--pdb_format', type=str, default='mmcif', help='pdb mmcif')
	parser.add_argument("--csv", required=True)
	parser.add_argument("--out", required=True)
	parser.add_argument("--chunksize", type=int, default=5000)
	parser.add_argument("--num_workers", type=int, default=-1, help="Number of worker processes for parallel processing (default: number of CPU cores - 2)")
	parser.add_argument("--resume", action="store_true", help="Resume from existing samples if available")
	parser.add_argument("--total_samples", type=int, default=0)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	out_path = Path(args.out)
	sample_dir = out_path.with_suffix("")
	sample_dir.mkdir(parents=True, exist_ok=True)

	rejection_keys = [
		"missing PDB file",
		"missing chain",
		"sequence mismatch",
		"invalid mutation position index",
		"invalid mutation position residue",
		"invalid mutation position PDB number",
		"not single mutation",
		"missing C-alpha",
		"failed alignment",
		"valid samples",
	]
	rejections = {key: [] for key in rejection_keys}
	rejections["valid samples"] = 0
	sample_ids: list[str] = []
	metadata: list[dict[str, Any]] = []

	total_rows = 0
	chunk_idx = 1
	max_workers = max(1, len(os.sched_getaffinity(0)) - 2) if args.num_workers < 0 else args.num_workers
	if max_workers == 1:
		for chunk in pd.read_csv(args.csv, chunksize=args.chunksize):
			total_rows += len(chunk)
			rows = chunk.to_dict(orient="records")
			
			for row in tqdm(rows, desc="prepare_data (debug)", leave=False):
				result = process_row_worker(row, args.pdb_format, args.pdb_root, args.pdb_version, sample_dir, args.resume)
				
				if result["status"] == "rejected":
					rejections[result["reason"]].append(result["sample_id"])
				else:
					rejections["valid samples"] += 1
					sample_ids.append(result["sample_id"])
					metadata.append(result["metadata"])
	else:
		with ProcessPoolExecutor(max_workers=max_workers) as executor:
			for chunk in pd.read_csv(args.csv, chunksize=args.chunksize):
				print(f"Processing chunk {chunk_idx} / {args.total_samples // args.chunksize + 1}")
				chunk_idx += 1
				total_rows += len(chunk)
				rows = chunk.to_dict(orient="records")
				futures = [executor.submit(process_row_worker, row, args.pdb_format, args.pdb_root, args.pdb_version, sample_dir, args.resume) for row in rows]
				for future in tqdm(as_completed(futures), total=len(futures), desc="prepare_data", leave=False):
					result = future.result()
					if result['status'] == "rejected":
						rejections[result['reason'].split(': ')[1]].append(result['sample_id'])
					else:
						rejections['valid samples'] += 1
						sample_ids.append(result['sample_id'])
						metadata.append(result['metadata'])

	manifest = {
		"format": "musrnet_manifest_v1",
		"samples_dir": str(sample_dir),
		"sample_ids": sample_ids,
		"metadata": metadata,
		"rejections": rejections,
		"total_rows": total_rows,
	}
	out_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(manifest, out_path)

	print("Rejection summary")
	for key in rejection_keys:
		print(f"{key}: {rejections[key]}")


if __name__ == "__main__":
	main()
