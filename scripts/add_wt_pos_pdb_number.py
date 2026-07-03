"""
python scripts/add_wt_pos_pdb_number.py --csv data/SingleMutPairs2024_subset_c1000.csv --out data/SingleMutPairs2024_subset_c1000.with_wt_pos.csv --pdb_root /rds/projects/l/liuje-multiai/shuo/datasets --pdb_version pdb_260603 --pdb_format mmcif --drop_bad
"""
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm
from Bio.SeqUtils import seq1 as three_to_one
from biopandas.pdb import PandasPdb
from biopandas.mmcif import PandasMmcif


ATOMS = [
    "C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3",
    "CG", "CG1", "CG2", "CH2", "CZ", "CZ2", "CZ3", "N", "ND1", "ND2",
    "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "O", "OD1", "OD2", "OE1",
    "OE2", "OG", "OG1", "OH", "OXT", "SD", "SG",
]

AA3_SET = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

VALID_ALT_LOCS = {"A", ""}

MMCIF_RENAME = {
    "auth_asym_id": "chain_id",
    "auth_seq_id": "residue_number",
    "auth_comp_id": "residue_name",
    "auth_atom_id": "atom_name",
    "label_alt_id": "alt_loc",
    "pdbx_PDB_ins_code": "insertion",
}


def get_structure_path(pdb_id, pdb_format, pdb_root, pdb_version):
    pdb_id = str(pdb_id).lower()

    if pdb_format == "mmcif":
        return Path(pdb_root) / pdb_version / "pdb" / f"{pdb_id}.cif.gz"

    if pdb_format == "pdb":
        return Path(pdb_root) / pdb_version / "pdb" / f"pdb{pdb_id}.ent.gz"

    raise ValueError(f"unknown pdb_format: {pdb_format}")


def normalize_insertion(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x in {"", ".", "?", "None", "nan"}:
        return ""
    return x


def load_atom_df(pdb_id, pdb_format, pdb_root, pdb_version):
    path = get_structure_path(pdb_id, pdb_format, pdb_root, pdb_version)

    if not path.exists():
        raise FileNotFoundError(str(path))

    if pdb_format == "mmcif":
        pdb = PandasMmcif().read_mmcif(str(path))
        atom_df = pdb.df["ATOM"].rename(columns=MMCIF_RENAME).copy()
        atom_df["residue_number"] = pd.to_numeric(atom_df["residue_number"], errors="coerce")
        return atom_df

    pdb = PandasPdb().read_pdb(str(path))
    return pdb.df["ATOM"].copy()


def filter_residues(atom_df, chain_id):
    df = atom_df[atom_df["chain_id"].astype(str) == str(chain_id)].copy()

    if df.empty:
        return df

    if "alt_loc" not in df.columns:
        df["alt_loc"] = ""

    df["alt_loc"] = (
        df["alt_loc"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({".": "", "?": "", "None": "", "nan": ""})
    )

    if "insertion" not in df.columns:
        df["insertion"] = ""

    df["insertion"] = df["insertion"].apply(normalize_insertion)

    df = df[
        df["atom_name"].astype(str).isin(ATOMS)
        & df["alt_loc"].isin(VALID_ALT_LOCS)
        & df["residue_name"].astype(str).str.upper().isin(AA3_SET)
        & df["residue_number"].notna()
    ].copy()

    residue_key_cols = ["residue_number", "insertion"]

    ca_residues = (
        df[df["atom_name"].astype(str) == "CA"][residue_key_cols]
        .drop_duplicates()
    )

    df = df.merge(ca_residues, on=residue_key_cols, how="inner")
    return df


def get_sequence_and_idx_mapping(atom_df, chain_id):
    df = filter_residues(atom_df, chain_id)

    if df.empty:
        raise ValueError("empty_or_missing_chain_after_filter")

    residue_key_cols = ["residue_number", "insertion"]

    unique_residues = (
        df.drop_duplicates(subset=residue_key_cols)
        .sort_values(by=residue_key_cols)
        [residue_key_cols + ["residue_name"]]
        .copy()
    )

    sequence = "".join(
        unique_residues["residue_name"]
        .astype(str)
        .str.upper()
        .apply(three_to_one)
        .tolist()
    )

    idx_to_pdb = {}

    for idx, row in enumerate(unique_residues.itertuples(index=False)):
        resnum = getattr(row, "residue_number")
        insertion = normalize_insertion(getattr(row, "insertion", ""))

        if pd.isna(resnum):
            continue

        resname = str(getattr(row, "residue_name")).upper()
        aa = three_to_one(resname)

        idx_to_pdb[int(idx)] = {
            "pdb_number": int(float(resnum)),
            "icode": insertion,
            "residue_name": resname,
            "aa": aa,
        }

    return sequence, idx_to_pdb


def _build_one_cache_item(task):
    """
    Worker for one WT chain.
    Input:
        (wt_pdb_id, wt_chain_id, pdb_format, pdb_root, pdb_version)
    Output:
        {
            "ok": bool,
            "key": (wt_pdb_id, wt_chain_id),
            "sequence": str | None,
            "idx_map": dict | None,
            "error": str | None,
        }
    """
    wt_pdb_id, wt_chain_id, pdb_format, pdb_root, pdb_version = task
    key = (str(wt_pdb_id).lower(), str(wt_chain_id))

    try:
        atom_df = load_atom_df(
            pdb_id=key[0],
            pdb_format=pdb_format,
            pdb_root=pdb_root,
            pdb_version=pdb_version,
        )

        seq, idx_map = get_sequence_and_idx_mapping(atom_df, key[1])

        return {
            "ok": True,
            "key": key,
            "sequence": seq,
            "idx_map": idx_map,
            "error": None,
        }

    except Exception as exc:
        return {
            "ok": False,
            "key": key,
            "sequence": None,
            "idx_map": None,
            "error": f"{type(exc).__name__}:{exc}",
        }


def build_cache_parallel(df, pdb_format, pdb_root, pdb_version, num_workers, chunksize):
    key_df = (
        df[["wt_pdb_id", "wt_chain_id"]]
        .drop_duplicates()
        .copy()
    )

    tasks = [
        (
            str(row.wt_pdb_id).lower(),
            str(row.wt_chain_id),
            pdb_format,
            str(pdb_root),
            str(pdb_version),
        )
        for row in key_df.itertuples(index=False)
    ]

    cache = {}
    chain_rejects = []

    if num_workers <= 1:
        iterator = map(_build_one_cache_item, tasks)
        results = tqdm(iterator, total=len(tasks), desc="Build WT chain cache")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(_build_one_cache_item, tasks, chunksize=chunksize)
            results = tqdm(results, total=len(tasks), desc="Build WT chain cache")

            for item in results:
                if item["ok"]:
                    cache[item["key"]] = {
                        "sequence": item["sequence"],
                        "idx_map": item["idx_map"],
                    }
                else:
                    chain_rejects.append({
                        "wt_pdb_id": item["key"][0],
                        "wt_chain_id": item["key"][1],
                        "reason": item["error"],
                    })

            return cache, chain_rejects

    for item in results:
        if item["ok"]:
            cache[item["key"]] = {
                "sequence": item["sequence"],
                "idx_map": item["idx_map"],
            }
        else:
            chain_rejects.append({
                "wt_pdb_id": item["key"][0],
                "wt_chain_id": item["key"][1],
                "reason": item["error"],
            })

    return cache, chain_rejects


def _fix_one_row(task):
    """
    Worker for one CSV row.
    Input:
        (row_dict, cache_item)
    Output:
        ("ok", fixed_row, None)
        or
        ("reject", None, reject_row)
    """
    row, cache_item = task

    sample_id = str(row["sample_id"])
    wt_pdb_id = str(row["wt_pdb_id"]).lower()
    wt_chain_id = str(row["wt_chain_id"])
    mut_pos = int(row["mut_pos_seq_index"])
    wt_sequence = str(row["wt_sequence"])
    wt_aa = str(row["wt_aa_type"])

    if cache_item is None:
        return (
            "reject",
            None,
            {
                "sample_id": sample_id,
                "reason": "wt_chain_cache_missing",
                "wt_pdb_id": wt_pdb_id,
                "wt_chain_id": wt_chain_id,
            },
        )

    try:
        parsed_seq = cache_item["sequence"]
        idx_map = cache_item["idx_map"]

        if parsed_seq != wt_sequence:
            return (
                "reject",
                None,
                {
                    "sample_id": sample_id,
                    "reason": "wt_sequence_mismatch",
                    "wt_pdb_id": wt_pdb_id,
                    "wt_chain_id": wt_chain_id,
                    "csv_len": len(wt_sequence),
                    "parsed_len": len(parsed_seq),
                },
            )

        if mut_pos not in idx_map:
            return (
                "reject",
                None,
                {
                    "sample_id": sample_id,
                    "reason": "mut_pos_seq_index_unmapped_in_wt",
                    "wt_pdb_id": wt_pdb_id,
                    "wt_chain_id": wt_chain_id,
                    "mut_pos_seq_index": mut_pos,
                },
            )

        mapped = idx_map[mut_pos]

        if mapped["aa"] != wt_aa:
            return (
                "reject",
                None,
                {
                    "sample_id": sample_id,
                    "reason": "wt_aa_mismatch_at_mut_pos",
                    "wt_pdb_id": wt_pdb_id,
                    "wt_chain_id": wt_chain_id,
                    "mut_pos_seq_index": mut_pos,
                    "csv_wt_aa": wt_aa,
                    "parsed_wt_aa": mapped["aa"],
                },
            )

        row = dict(row)
        row["wt_pos_pdb_number"] = int(mapped["pdb_number"])
        row["wt_pos_pdb_icode"] = str(mapped["icode"])

        return ("ok", row, None)

    except Exception as exc:
        return (
            "reject",
            None,
            {
                "sample_id": sample_id,
                "reason": f"exception:{type(exc).__name__}:{exc}",
                "wt_pdb_id": wt_pdb_id,
                "wt_chain_id": wt_chain_id,
            },
        )


def add_wt_pos_pdb_number_parallel(df, cache, num_workers, chunksize):
    required = [
        "sample_id",
        "wt_pdb_id",
        "wt_chain_id",
        "mut_pos_seq_index",
        "wt_aa_type",
        "wt_sequence",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    records = df.to_dict(orient="records")

    tasks = []
    for row in records:
        key = (str(row["wt_pdb_id"]).lower(), str(row["wt_chain_id"]))
        tasks.append((row, cache.get(key)))

    fixed_rows = []
    reject_rows = []

    if num_workers <= 1:
        iterator = map(_fix_one_row, tasks)
        results = tqdm(iterator, total=len(tasks), desc="Add wt_pos_pdb_number")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(_fix_one_row, tasks, chunksize=chunksize)
            results = tqdm(results, total=len(tasks), desc="Add wt_pos_pdb_number")

            for status, fixed_row, reject_row in results:
                if status == "ok":
                    fixed_rows.append(fixed_row)
                else:
                    reject_rows.append(reject_row)

            return pd.DataFrame(fixed_rows), pd.DataFrame(reject_rows)

    for status, fixed_row, reject_row in results:
        if status == "ok":
            fixed_rows.append(fixed_row)
        else:
            reject_rows.append(reject_row)

    return pd.DataFrame(fixed_rows), pd.DataFrame(reject_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rejects_out", default=None)
    parser.add_argument("--chain_rejects_out", default=None)
    parser.add_argument("--pdb_root", default="/rds/projects/l/liuje-multiai/shuo/datasets")
    parser.add_argument("--pdb_version", default="pdb_260603")
    parser.add_argument("--pdb_format", default="mmcif", choices=["mmcif", "pdb"])
    parser.add_argument("--drop_bad", action="store_true")

    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, len(os.sched_getaffinity(0)) - 2),
    )
    parser.add_argument("--chunksize_cache", type=int, default=1)
    parser.add_argument("--chunksize_rows", type=int, default=500)

    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)

    df["wt_pdb_id"] = df["wt_pdb_id"].astype(str).str.lower()
    df["mut_pdb_id"] = df["mut_pdb_id"].astype(str).str.lower()
    df["wt_chain_id"] = df["wt_chain_id"].astype(str)
    df["mut_chain_id"] = df["mut_chain_id"].astype(str)
    df["mut_pos_seq_index"] = pd.to_numeric(
        df["mut_pos_seq_index"],
        errors="raise",
    ).astype(int)

    cache, chain_rejects = build_cache_parallel(
        df=df,
        pdb_format=args.pdb_format,
        pdb_root=args.pdb_root,
        pdb_version=args.pdb_version,
        num_workers=args.num_workers,
        chunksize=args.chunksize_cache,
    )

    fixed_df, row_rejects_df = add_wt_pos_pdb_number_parallel(
        df=df,
        cache=cache,
        num_workers=args.num_workers,
        chunksize=args.chunksize_rows,
    )

    chain_rejects_df = pd.DataFrame(chain_rejects)

    all_rejects = []
    if not chain_rejects_df.empty:
        chain_rejects_df = chain_rejects_df.copy()
        chain_rejects_df["sample_id"] = ""
        all_rejects.append(chain_rejects_df)

    if not row_rejects_df.empty:
        all_rejects.append(row_rejects_df)

    rejects_df = pd.concat(all_rejects, ignore_index=True, sort=False) if all_rejects else pd.DataFrame()

    if not args.drop_bad and not rejects_df.empty:
        rejects_out = args.rejects_out
        if rejects_out is None:
            rejects_out = str(args.out).replace(".csv", ".rejects.csv")

        Path(rejects_out).parent.mkdir(parents=True, exist_ok=True)
        rejects_df.to_csv(rejects_out, index=False)

        raise RuntimeError(
            f"{len(rejects_df)} rows/chains failed mapping. "
            f"Inspect rejects file: {rejects_out}. "
            f"Use --drop_bad to write only valid rows."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fixed_df.to_csv(out, index=False)

    rejects_out = args.rejects_out
    if rejects_out is None:
        rejects_out = str(out).replace(".csv", ".rejects.csv")

    chain_rejects_out = args.chain_rejects_out
    if chain_rejects_out is None:
        chain_rejects_out = str(out).replace(".csv", ".chain_rejects.csv")

    rejects_df.to_csv(rejects_out, index=False)
    chain_rejects_df.to_csv(chain_rejects_out, index=False)

    print(f"input rows: {len(df)}")
    print(f"written rows: {len(fixed_df)}")
    print(f"row/chain rejected rows: {len(rejects_df)}")
    print(f"unique WT chains cached: {len(cache)}")
    print(f"chain rejects: {len(chain_rejects_df)}")
    print(f"output: {out}")
    print(f"rejects: {rejects_out}")
    print(f"chain rejects: {chain_rejects_out}")

    if "mut_pos_pdb_number" in fixed_df.columns and not fixed_df.empty:
        same = (
            pd.to_numeric(fixed_df["mut_pos_pdb_number"], errors="coerce")
            == pd.to_numeric(fixed_df["wt_pos_pdb_number"], errors="coerce")
        )
        print(f"wt_pos_pdb_number == mut_pos_pdb_number: {int(same.sum())}/{len(same)}")
        print(f"wt_pos_pdb_number != mut_pos_pdb_number: {int((~same).sum())}/{len(same)}")


if __name__ == "__main__":
    main()