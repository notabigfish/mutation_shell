"""
python scripts/fetch_domain_annotations.py \
  --sample-csv data/SingleMutPairs2024.csv \
  --out-csv data/domain_annotations.csv \
  --cache-json data/cache/pdbe_sifts_domain_cache.json \
  --sources CATH,Pfam,SCOP,InterPro  # CATH,SCOP for strict structural domains only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

REQUIRED_SAMPLE_COLUMNS = [
    "sample_id",
    "wt_pdb_id",
    "wt_chain_id",
    "mut_pdb_id",
    "mut_chain_id",
]

PDBe_ENDPOINTS = {
    "CATH": "https://www.ebi.ac.uk/pdbe/api/mappings/cath/{pdb_id}",
    "SCOP": "https://www.ebi.ac.uk/pdbe/api/mappings/scop/{pdb_id}",
    "Pfam": "https://www.ebi.ac.uk/pdbe/api/mappings/pfam/{pdb_id}",
    "InterPro": "https://www.ebi.ac.uk/pdbe/api/mappings/interpro/{pdb_id}",
}

def normalize_pdb_id(pdb_id: str) -> str:
    return str(pdb_id).strip().lower()

def normalize_chain_id(chain_id: str) -> str:
    return str(chain_id).strip()

def validate_sample_csv(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_SAMPLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in sample CSV: {missing}")

def request_json(
    url: str,
    timeout: int = 30,
    retries: int = 3,
    sleep_s: float = 0.5,
    session: requests.Session | None = None,
) -> dict[str, Any] | None:
    client = session if session is not None else requests

    for attempt in range(retries):
        try:
            response = client.get(url, timeout=timeout)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt + 1 == retries:
                return None
            time.sleep(sleep_s * (attempt + 1))

    return None

def load_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache: dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f)
    tmp_path.replace(cache_path)

def get_residue_number(position_obj: Any) -> int | None:
    """
    PDBe SIFTS mapping position objects:
        {
          "residue_number": 10,
          "author_residue_number": 15,
          "author_insertion_code": ""
        }

    author_residue_number is usually closer to PDB residue numbering.
    Fall back to residue_number.
    """
    if not isinstance(position_obj, dict):
        print(f"Warning: Unexpected position object format: {position_obj}")
        return None

    for key in ["author_residue_number", "residue_number"]:
        value = position_obj.get(key)
        if value is None:
            print(f"Warning: Missing {key} in position object: {position_obj}")
            continue
        try:
            return int(value)
        except Exception as e:
            print(f"Warning: Failed to convert {key} value to int in position object: {position_obj}. Error: {e}")
            continue

    return None

def extract_rows_from_pdbe_mapping(pdb_id, source_name, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """
    Parse PDBe SIFTS mapping response

    Expected broad shape:
    {
        "1abc": {
            "CATH": {
                "domain_accession": {
                    "identifier": "...",
                    "name": "...",
                    "mappings": [
                        {
                            "chain_id": "A",
                            "start": {...},
                            "end": {...},
                        }
                    ]
                }
            }
        }
    }
    """
    if not payload:
        print(f"Warning: Empty payload for PDB ID {pdb_id} and source {source_name}")
        return []

    pdb_key = pdb_id.lower()
    if pdb_key not in payload:
        pdb_key = pdb_id.upper()
    if pdb_key not in payload:
        print(f"Warning: PDB ID {pdb_id} not found in PDBe SIFTS payload for source {source_name}")
        return []
    entry = payload[pdb_key]
    if not isinstance(entry, dict):
        print(f"Warning: Unexpected entry format for PDB ID {pdb_id} in PDBe SIFTS payload for source {source_name}")
        return []

    source_block = None
    for possible_key in [source_name, source_name.lower(), source_name.upper()]:
        if possible_key in entry:
            source_block = entry[possible_key]
            break
    if not isinstance(source_block, dict): 
        print(f"Warning: Unexpected source block format for PDB ID {pdb_id} in PDBe SIFTS payload for source {source_name}")
        return []
    
    rows: list[dict[str, Any]] = []
    for accession, annotation in source_block.items():
        if not isinstance(annotation, dict):
            print(f"Warning: Unexpected annotation format for {pdb_id} {source_name}: {annotation}")
            continue
        mappings = annotation.get("mappings", [])
        if not isinstance(mappings, list): continue

        for map_idx, mapping in enumerate(mappings):
            if not isinstance(mapping, dict):
                print(f"Warning: Unexpected mapping format for {pdb_id} {source_name} {accession}: {mapping}")
                continue
            chain_id = (
                mapping.get("chain_id")
                or mapping.get("author_chain_id")
                or mapping.get("struct_asym_id")
            )
            if chain_id is None:
                print(f"Warning: Missing chain_id for {pdb_id} {source_name} {accession} mapping {map_idx}")
                continue
            start = get_residue_number(mapping.get("start"))
            end = get_residue_number(mapping.get("end"))
            if start is None or end is None:
                print(f"Warning: Missing start/end for {pdb_id} {source_name} {accession} mapping {map_idx}")
                continue
            if end < start: start, end = end, start

            domain_id = f"{source_name}:{accession}:{chain_id}:{start}-{end}:{map_idx}"

            rows.append({
                "pdb_id": pdb_id,
                "chain_id": str(chain_id),
                "domain_id": domain_id,
                "domain_source": source_name,
                "domain_start": int(start),
                "domain_end": int(end),
            })
    return rows

def fetch_domains_for_pdb(
    pdb_id: str,
    sources: list[str],
    cache: dict[str, Any],
    sleep_s: float,
    cache_lock: Lock,
) -> list[dict[str, Any]]:
    pdb_id = normalize_pdb_id(pdb_id)
    all_rows: list[dict[str, Any]] = []

    with requests.Session() as session:
        for source in sources:
            cache_key = f"{source}:{pdb_id}"

            with cache_lock:
                cached = cache_key in cache
                payload = cache.get(cache_key)

            if not cached:
                url = PDBe_ENDPOINTS[source].format(pdb_id=pdb_id)
                payload = request_json(url, session=session)

                with cache_lock:
                    cache[cache_key] = payload

                time.sleep(sleep_s)

            rows = extract_rows_from_pdbe_mapping(
                pdb_id=pdb_id,
                source_name=source,
                payload=payload,
            )
            all_rows.extend(rows)

    return all_rows

def deduplicate_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "pdb_id",
                "chain_id",
                "domain_source",
                "domain_id",
                "domain_start",
                "domain_end",
            ]
        )

    df = pd.DataFrame(rows)
    df["pdb_id"] = df["pdb_id"].astype(str).str.lower()
    df["chain_id"] = df["chain_id"].astype(str)
    df["domain_source"] = df["domain_source"].astype(str)
    df["domain_id"] = df["domain_id"].astype(str)
    df["domain_start"] = pd.to_numeric(df["domain_start"], errors="coerce").astype("Int64")
    df["domain_end"] = pd.to_numeric(df["domain_end"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["domain_start", "domain_end"]).copy()
    df["domain_start"] = df["domain_start"].astype(int)
    df["domain_end"] = df["domain_end"].astype(int)

    df = df.drop_duplicates(
        subset=[
            "pdb_id",
            "chain_id",
            "domain_source",
            "domain_id",
            "domain_start",
            "domain_end",
        ]
    )

    df = df.sort_values(
        ["pdb_id", "chain_id", "domain_source", "domain_start", "domain_end"]
    ).reset_index(drop=True)

    return df

def build_domain_annotations(
    sample_csv: Path,
    out_csv: Path,
    cache_json: Path,
    sources: list[str],
    sleep_s: float,
    max_samples: int = -1,
    max_workers: int = 8
):
    sample_df = pd.read_csv(sample_csv)
    validate_sample_csv(sample_df)

    pdb_ids = sorted({
        normalize_pdb_id(x)
        for x in sample_df["wt_pdb_id"].dropna().unique().tolist()
    })
    if max_samples > 0:
        pdb_ids = pdb_ids[:max_samples]

    cache = load_cache(cache_json)

    rows: list[dict[str, Any]] = []
    empty_anno_pdbids = []
    cache_lock = Lock()
    max_workers = min(max_workers, max(1, len(pdb_ids)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fetch_domains_for_pdb,
                pdb_id,
                sources,
                cache,
                sleep_s,
                cache_lock,
            ): pdb_id
            for pdb_id in pdb_ids
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fetching PDBe SIFTS domain mappings",
        ):
            pdb_id = futures[future]
            out_rows = future.result()
            rows.extend(out_rows)

            if out_rows == []:
                empty_anno_pdbids.append(pdb_id)

    save_cache(cache, cache_json)

    domain_df = deduplicate_rows(rows)

    needed_chains = set(zip(
        sample_df["wt_pdb_id"].map(normalize_pdb_id),
        sample_df["wt_chain_id"].map(normalize_chain_id),
    ))

    if not domain_df.empty:
        domain_df = domain_df[
            domain_df.apply(
                lambda r: (normalize_pdb_id(r["pdb_id"]), normalize_chain_id(r["chain_id"])) in needed_chains,
                axis=1
            )
        ].copy()
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    domain_df.to_csv(out_csv, index=False)

    n_pdb = len(pdb_ids)
    n_rows = len(domain_df)
    n_chains = domain_df[['pdb_id', 'chain_id']].drop_duplicates().shape[0] if n_rows else 0

    print(f"PDB IDs with domain annotations: {n_pdb - len(empty_anno_pdbids)} / {n_pdb}")
    print(f"Domain rows written: {n_rows}")
    print(f"Annotated WT chains: {n_chains}")
    print(f"Output: {out_csv}")
    print(f"Cache: {cache_json}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch domain annotations for MuSRNet samples using PDBe SIFTS")
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--out-csv", default="data/domain_annotations.csv")
    parser.add_argument("--cache-json", default="data/cache/pdbe_sifts_domain_cache.json")
    parser.add_argument(
        "--sources",
        default="CATH,Pfam,SCOP,InterPro",
        help="Comma-separated domain sources. Recommended: CATH,Pfam,SCOP,InterPro",
    )
    parser.add_argument("--sleep-s", type=float, default=0.1)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=-1, help="For testing: limit number of samples to process")
    return parser.parse_args()

def main():
    args = parse_args()

    sources = [x.strip() for x in args.sources.split(",") if x.strip()]
    invalid = [x for x in sources if x not in PDBe_ENDPOINTS]
    if invalid:
        raise ValueError(f"Invalid sources: {invalid}. Valid sources: {sorted(PDBe_ENDPOINTS)}")
    
    build_domain_annotations(
        sample_csv=Path(args.sample_csv),
        out_csv=Path(args.out_csv),
        cache_json=Path(args.cache_json),
        sources=sources,
        sleep_s=float(args.sleep_s),
        max_samples=int(args.max_samples),
        max_workers=int(args.max_workers),
    )
    
if __name__ == "__main__":
    main()