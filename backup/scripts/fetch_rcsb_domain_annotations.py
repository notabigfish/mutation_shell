from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_COLUMNS = ["pdb_id", "chain_id", "domain_source", "domain_id", "domain_start", "domain_end"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best-effort RCSB domain annotation fetcher")
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--cache-json", required=True)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit the number of samples to process (for testing)")
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_cache(path: Path, cache: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2)


def fetch_entry_annotations(pdb_id: str) -> list[dict[str, object]]:
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{pdb_id.lower()}/A"
    request = urllib.request.Request(url, headers={"User-Agent": "MuSRNet/1.0"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    annotations = payload.get("rcsb_polymer_instance_annotation", [])
    rows: list[dict[str, object]] = []
    for item in annotations:
        source = item.get("provenance_source") or item.get("annotation_lineage", [{}])[0].get("name") or "RCSB"
        ann_id = item.get("annotation_id") or item.get("name")
        for region in item.get("annotation_regions", []):
            start = region.get("begin_sequence_number")
            end = region.get("end_sequence_number")
            if ann_id is None or start is None or end is None:
                continue
            rows.append(
                {
                    "pdb_id": pdb_id.lower(),
                    "chain_id": "A",
                    "domain_source": str(source),
                    "domain_id": str(ann_id),
                    "domain_start": int(start),
                    "domain_end": int(end),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    sample_df = pd.read_csv(args.sample_csv)
    out_csv = Path(args.out_csv)
    cache_json = Path(args.cache_json)
    cache = load_cache(cache_json)
    if args.max_samples is not None:
        sample_df = sample_df.head(args.max_samples)
    unique_pairs = sample_df[["wt_pdb_id", "wt_chain_id"]].drop_duplicates().to_dict(orient="records")
    rows: list[dict[str, object]] = []
    warnings: list[str] = []

    for item in unique_pairs:
        pdb_id = str(item["wt_pdb_id"]).lower()
        chain_id = str(item["wt_chain_id"])
        cache_key = f"{pdb_id}:{chain_id}"
        print(cache_key)
        if cache_key in cache:
            cached_rows = cache[cache_key]
            if isinstance(cached_rows, list):
                rows.extend(cached_rows)
            continue
        try:
            fetched = fetch_entry_annotations(pdb_id)
            fetched = [dict(row, chain_id=chain_id) for row in fetched]
            cache[cache_key] = fetched
            rows.extend(fetched)
        except Exception as exc:
            warnings.append(f"domain annotation fetch failed for {cache_key}: {exc}")
            cache[cache_key] = []

    save_cache(cache_json, cache)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out_csv, index=False)

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(warning)
    if not rows:
        print("No domain annotations were fetched. Wrote empty CSV with expected columns.")


if __name__ == "__main__":
    main()
