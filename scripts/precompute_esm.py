from __future__ import annotations
"""
python scripts/precompute_esm.py --samples data/processed/samples_subset_c50_raw.pt --out-esm-lmdb data/processed/esm_subset_c50.lmdb --out-filtered-manifest data/processed/samples_subset_c50.pt
"""
import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from musrnet.dataset import iter_sample_paths, load_samples_manifest
from musrnet.esm_embed import ESMEmbedder, save_chain_esm, open_esm_lmdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute frozen ESM embeddings")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--model-name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--max-len", type=int, default=1022)
    parser.add_argument("--out-filtered-manifest", type=str, default=None)
    parser.add_argument("--out-esm-lmdb", type=str, default=None, help="Path to LMDB database for storing embeddings (optional)")
    return parser.parse_args()

def chain_key(pdb_id: str, chain_id: str) -> str:
    return f"{str(pdb_id).lower()}_{str(chain_id)}"

def main() -> None:
    args = parse_args()
    manifest = load_samples_manifest(args.samples)
    lmdb_path = args.out_esm_lmdb
    env = open_esm_lmdb(lmdb_path, readonly=False) if lmdb_path else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = ESMEmbedder(args.model_name, device=device)

    kept_sample_ids = []
    kept_metadata = []
    metadata_by_id = {item['sample_id']: item for item in manifest.get("metadata", [])}

    skipped = 0
    processed = 0
    cached_chain_ids = set()
    
    with env:
        for sample_path in tqdm(iter_sample_paths(manifest), total=len(manifest["sample_ids"]), desc="precompute_esm"):
            sample = torch.load(sample_path, map_location="cpu")
            if len(sample["wt_sequence"]) > args.max_len or len(sample["mut_sequence"]) > args.max_len:
                skipped += 1
                continue
            wt_key = chain_key(sample["wt_pdb_id"], sample["wt_chain_id"])
            mut_key = chain_key(sample["mut_pdb_id"], sample["mut_chain_id"])
            with env.begin(write=False) as txn:
                wt_exists = txn.get(wt_key.encode("utf-8")) is not None
                mut_exists = txn.get(mut_key.encode("utf-8")) is not None

            if wt_key not in cached_chain_ids and not wt_exists:
                wt_embedding = embedder.embed_sequence(sample["wt_sequence"])
                save_chain_esm(
                    env,
                    key=wt_key,
                    pdb_id=sample["wt_pdb_id"],
                    chain_id=sample["wt_chain_id"],
                    sequence=sample["wt_sequence"],
                    embedding=wt_embedding)
            cached_chain_ids.add(wt_key)
            
            if mut_key not in cached_chain_ids and not mut_exists:
                mut_embedding = embedder.embed_sequence(sample["mut_sequence"])
                save_chain_esm(
                    env,
                    key=mut_key,
                    pdb_id=sample["mut_pdb_id"],
                    chain_id=sample["mut_chain_id"],
                    sequence=sample["mut_sequence"],
                    embedding=mut_embedding)
            cached_chain_ids.add(mut_key)

            kept_sample_ids.append(sample["sample_id"])
            if sample["sample_id"] in metadata_by_id:
                kept_metadata.append(metadata_by_id[sample["sample_id"]])
            processed += 1

    filtered_manifest = {
        **manifest,
        "sample_ids": kept_sample_ids,
        "metadata": kept_metadata,
        "source_manifest": str(Path(args.samples).resolve()),
        "esm_model_name": args.model_name,
        "esm_max_len": args.max_len,
        "esm_key_type": "pdb_id_chain_id",
        "esm_storage": "lmdb",
        "esm_lmdb_path": lmdb_path,
        "esm_dtype": "float16",        
    }
    filtered_manifest_path = args.out_filtered_manifest
    torch.save(filtered_manifest, filtered_manifest_path)
    print(f"Processed samples: {processed}")
    print(f"Skipped for length > {args.max_len}: {skipped}")


if __name__ == "__main__":
    main()
