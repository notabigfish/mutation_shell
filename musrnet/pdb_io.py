from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.SeqUtils import seq1 as three_to_one
from process_data import read_pdb, load_structure_file, get_seq_and_mapping, filter_residues, AA3_SET, VALID_ALT_LOCS

def find_pdb_file(pdb_dir: str, pdb_id: str) -> Path:
    pdb_path = Path(pdb_dir) / f"{pdb_id.lower()}.cif.gz"
    if not pdb_path.exists():
        raise FileNotFoundError(f"Missing PDB file: {pdb_path}")
    return pdb_path

def _pdb_id_from_path(pdb_path: Path) -> str:
    return pdb_path.name.split(".", 1)[0].lower()


def _residue_key_columns(df: pd.DataFrame) -> list[str]:
    key_cols = ["residue_number"]
    if "insertion" in df.columns:
        key_cols.append("insertion")
    return key_cols


def _clean_insertion(value: Any) -> str:
    if pd.isna(value) or value in {"?", ".", "None", "nan"}:
        return ""
    return str(value).strip()


def load_chain(pdb_path: Path, chain_id: str, pdb_format: str, pdb_root: str, pdb_version: str) -> pd.DataFrame:
    chain_df = read_pdb(_pdb_id_from_path(pdb_path), chain_id, pdb_format, pdb_root, pdb_version)
    if chain_df.empty:
        raise KeyError(f"Missing chain {chain_id} in {pdb_path.name}")
    return chain_df


def inspect_chain(pdb_path: Path, chain_id: str, pdb_format: str, pdb_root: str, pdb_version: str) -> dict[str, Any]:
    pdb_id = _pdb_id_from_path(pdb_path)
    try:
        structure, _ = load_structure_file(pdb_id, pdb_format, pdb_root, pdb_version)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing PDB file: {pdb_path}") from exc
    atom_df = structure.df["ATOM"]
    chain_atom_df = atom_df[atom_df["chain_id"] == chain_id].copy()
    if chain_atom_df.empty:
        raise KeyError(f"Missing chain {chain_id} in {pdb_path.name}")

    filtered_df = filter_residues(atom_df, chain_id=chain_id)
    if filtered_df.empty:
        raise KeyError(f"Missing chain {chain_id} in {pdb_path.name}")

    sequence, _ = get_seq_and_mapping(filtered_df)
    key_cols = _residue_key_columns(chain_atom_df)
    standard_residue_df = chain_atom_df.copy()
    if "alt_loc" not in standard_residue_df.columns:
        standard_residue_df["alt_loc"] = ""
    standard_residue_df["alt_loc"] = (
        standard_residue_df["alt_loc"].fillna("").astype(str).str.strip().replace({".": "", "?": "", "None": ""})
    )
    standard_residue_df = standard_residue_df[
        standard_residue_df["residue_name"].isin(AA3_SET)
        & standard_residue_df["alt_loc"].isin(VALID_ALT_LOCS)
    ].copy()
    all_residue_keys = {
        tuple(row)
        for row in standard_residue_df[key_cols].drop_duplicates().itertuples(index=False, name=None)
    }
    modeled_residue_keys = {
        tuple(row) for row in filtered_df[key_cols].drop_duplicates().itertuples(index=False, name=None)
    }
    missing_ca_count = len(all_residue_keys - modeled_residue_keys)

    ca_df = filtered_df[filtered_df["atom_name"] == "CA"].copy()
    ca_df = ca_df.drop_duplicates(subset=key_cols).sort_values(by=key_cols)
    coords = ca_df[['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy(dtype=np.float32)
    residues = [
        {
            "pdb_number": int(row.residue_number),
            "icode": _clean_insertion("" if pd.isna(row.insertion) else str(row.insertion)),
            "aa": three_to_one(row.residue_name),
        }
        for row in ca_df[key_cols + ["residue_name"]].itertuples(index=False)
    ]

    return {
        "sequence": sequence,
        "coords": coords,
        "residues": residues,
        "missing_ca_count": missing_ca_count,
        "chain_df": filtered_df,
    }


def extract_ca_sequence_and_coords(pdb_path: Path, chain_id: str, pdb_format: str, pdb_root: str, pdb_version: str) -> tuple[str, np.ndarray, list]:
    info = inspect_chain(pdb_path, chain_id, pdb_format, pdb_root, pdb_version)
    return info["sequence"], info["coords"], info["residues"]
