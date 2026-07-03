from __future__ import annotations

import gzip
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import MMCIF2Dict, MMCIFParser
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from musrnet.pdb_io import find_pdb_file
from musrnet.statistics import paired_bootstrap_ci, wilcoxon_signed_rank_pvalue
import freesasa
freesasa.setVerbosity(freesasa.nowarnings)
from Bio.Data.PDBData import protein_letters_1to3
from Bio.PDB.DSSP import make_dssp_dict, residue_max_acc
from Bio.PDB.DSSP import DSSP
import concurrent.futures
from tqdm import tqdm
import copy
import re

AA_VOLUME = {
    "G": 60.1,
    "A": 88.6,
    "S": 89.0,
    "C": 108.5,
    "D": 111.1,
    "P": 112.7,
    "N": 114.1,
    "T": 116.1,
    "E": 138.4,
    "V": 140.0,
    "Q": 143.8,
    "H": 153.2,
    "M": 162.9,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "R": 173.4,
    "F": 189.9,
    "Y": 193.6,
    "W": 227.8,
}
POSITIVE = {"K", "R", "H"}
NEGATIVE = {"D", "E"}
CHARGED = POSITIVE | NEGATIVE
MAX_ASA = {
    "A": 121.0, "R": 265.0, "N": 187.0, "D": 187.0, "C": 148.0, "Q": 214.0, "E": 214.0, "G": 97.0,
    "H": 216.0, "I": 195.0, "L": 191.0, "K": 230.0, "M": 203.0, "F": 228.0, "P": 154.0, "S": 143.0,
    "T": 163.0, "W": 264.0, "Y": 255.0, "V": 165.0,
}
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
}

PREDICTION_COLUMN_ALIASES = {
    "sample_id": ["sample_id", "protein_id"],
    "cluster_id_30": ["cluster_id_30", "cluster_id", "cluster"],
    "shell_id": ["shell_id", "shell"],
    "true_displacement": ["true_displacement", "true_disp", "y_disp"],
    "pred_displacement": ["pred_displacement", "pred_disp", "disp"],
    "true_perturbed": ["true_perturbed", "y_perturbed"],
    "pred_perturbed_prob": ["pred_perturbed_prob", "perturbed_prob", "pred_prob"],
    "radii": ["radii"],
    "true_radius": ["true_radius"],
    "pred_radius": ["pred_radius"],
    "true_class": ["true_class"],
    "pred_class": ["pred_class"],
    "wt_aa_type": ["wt_aa_type", "wt_aa"],
    "mut_aa_type": ["mut_aa_type", "mut_aa"],
    "mut_pos_seq_index": ["mut_pos_seq_index", "mut_pos"],
    "wt_pdb_id": ["wt_pdb_id"],
    "wt_chain_id": ["wt_chain_id"],
    "mut_pdb_id": ["mut_pdb_id"],
    "mut_chain_id": ["mut_chain_id"],
    "release_date": ["release_date"],
}
REQUIRED_PREDICTION_COLUMNS = [
    "sample_id",
    "cluster_id_30",
    "shell_id",
    "true_displacement",
    "pred_displacement",
    "true_perturbed",
    "pred_perturbed_prob",
]
REQUIRED_SAMPLE_COLUMNS = [
    "sample_id",
    "wt_pdb_id",
    "wt_chain_id",
    "mut_pdb_id",
    "mut_chain_id",
    "mut_pos_seq_index",
    "mut_pos_pdb_number",
    "wt_aa_type",
    "mut_aa_type",
    "wt_sequence",
    "mut_sequence",
    "cluster_id_30",
    "release_date",
]
LOWER_IS_BETTER_METRICS = {"global_mae", "shell_mae", "mae_shell_0", "mae_shell_1", "mae_shell_2", "mae_shell_3", "mae_shell_4", "radius_mae"}
HIGHER_IS_BETTER_METRICS = {"perturbed_auroc", "perturbed_auprc", "class_correct", "class_macro_f1", "non_local_f1", "non_local_recall"}


def _warn(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for normalized, aliases in PREDICTION_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = normalized
                break
    df = df.rename(columns=rename_map).copy()
    missing = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    for column in ["sample_id", "cluster_id_30", "wt_pdb_id", "wt_chain_id", "mut_pdb_id", "mut_chain_id", "wt_aa_type", "mut_aa_type", "release_date"]:
        if column in df.columns:
            df[column] = df[column].astype(str)

    numeric_columns = [
        "shell_id",
        "true_displacement",
        "pred_displacement",
        "true_perturbed",
        "pred_perturbed_prob",
        "radii",
        "true_radius",
        "pred_radius",
        "true_class",
        "pred_class",
        "mut_pos_seq_index",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=REQUIRED_PREDICTION_COLUMNS).copy()
    if df.empty:
        raise ValueError("Empty prediction table after normalizing required columns")
    df["shell_id"] = df["shell_id"].astype(int)
    df["true_perturbed"] = df["true_perturbed"].astype(int)
    return df


def assign_substitution_size_group(wt_aa: str, mut_aa: str) -> str:
    delta = AA_VOLUME[mut_aa] - AA_VOLUME[wt_aa]
    if delta > 0:
        return "small_to_large"
    if delta < 0:
        return "large_to_small"
    return "size_neutral"


def assign_charge_group(wt_aa: str, mut_aa: str) -> str:
    if wt_aa in CHARGED and mut_aa not in CHARGED:
        return "charged_to_neutral"
    if wt_aa not in CHARGED and mut_aa in CHARGED:
        return "neutral_to_charged"
    if wt_aa in CHARGED and mut_aa in CHARGED:
        if wt_aa in POSITIVE and mut_aa in NEGATIVE:
            return "positive_to_negative"
        if wt_aa in NEGATIVE and mut_aa in POSITIVE:
            return "negative_to_positive"
        return "charged_to_charged_same_sign"
    return "neutral_to_neutral"


def assign_glypro_group(wt_aa: str, mut_aa: str) -> str:
    return "gly_or_pro_involved" if wt_aa in {"G", "P"} or mut_aa in {"G", "P"} else "no_gly_or_pro"


def assign_length_group(length: int, threshold: float) -> str:
    return "short" if length <= threshold else "long"


def assign_resolution_group(resolution: float, high_threshold: float) -> str:
    if pd.isna(resolution):
        return "unknown"
    return "high_resolution" if float(resolution) <= high_threshold else "low_resolution"


def assign_secondary_structure_group(dssp_code: str) -> str:
    if dssp_code in {"H", "G", "I"}:
        return "helix"
    if dssp_code in {"E", "B"}:
        return "sheet"
    if dssp_code in {"T", "S", "-"}:
        return "loop"
    return "unknown"


def assign_exposure_group(rsa: float, rsa_threshold: float) -> str:
    if pd.isna(rsa):
        return "unknown"
    return "buried" if float(rsa) < rsa_threshold else "exposed"

DOMAIN_COLUMNS = [
    "pdb_id",
    "chain_id",
    "domain_source",
    "domain_id",
    "domain_start",
    "domain_end",
]

def load_domain_annotations(path: str | Path | None, warnings: list[str]) -> pd.DataFrame:
    if path is None:
        _warn(warnings, "No domain annotations provided; using empty DataFrame")
        return pd.DataFrame(columns=DOMAIN_COLUMNS)
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Domain annotations file not found: {path}")
    
    df = pd.read_csv(path)
    missing = [c for c in DOMAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Domain annotation file missing columns: {missing}")
    
    df["pdb_id"] = df["pdb_id"].astype(str).str.lower()
    df["chain_id"] = df["chain_id"].astype(str)
    df["domain_source"] = df["domain_source"].astype(str)
    df["domain_id"] = df["domain_id"].astype(str)
    df["domain_start"] = pd.to_numeric(df["domain_start"], errors="coerce")
    df["domain_end"] = pd.to_numeric(df["domain_end"], errors="coerce")

    df = df.dropna(subset=["domain_start", "domain_end"]).copy()
    df["domain_start"] = df["domain_start"].astype(int)
    df["domain_end"] = df["domain_end"].astype(int)

    return df

def assign_domain_group_for_samples(
    sample_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    preferred_sources: tuple[str, ...] = ("CATH", "SCOP", "Pfam", "InterPro"),
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """
    Adds:
        domain_count
        domain_group
        domain_source_used

    Rule:
        use first available source in preferred_sources.
        CATH > SCOP > Pfam > InterPro.
    """
    warnings = warnings if warnings is not None else []
    sample_df = sample_df.copy()

    required = ["sample_id", "wt_pdb_id", "wt_chain_id"]
    missing = [c for c in required if c not in sample_df.columns]
    if missing:
        raise ValueError(f"sample_df missing columns: {missing}")

    if domain_df.empty:
        _warn(warnings, "Empty domain annotation DataFrame; all samples will have unknown domain group")
        sample_df["domain_count"] = np.nan
        sample_df["domain_group"] = "unknown"
        sample_df["domain_source_used"] = "none"
        return sample_df

    domain_df = domain_df.copy()
    domain_df["pdb_id"] = domain_df["pdb_id"].astype(str).str.lower()
    domain_df["chain_id"] = domain_df["chain_id"].astype(str)
    domain_df["domain_source"] = domain_df["domain_source"].astype(str)

    # Backward compatibility only. Your fetch script writes "SCOP".
    domain_df["domain_source"] = domain_df["domain_source"].replace({"SCOPe": "SCOP"})

    sample_df["_pdb_id_norm"] = sample_df["wt_pdb_id"].astype(str).str.lower()
    sample_df["_chain_id_norm"] = sample_df["wt_chain_id"].astype(str)

    source_rank = {source: i for i, source in enumerate(preferred_sources)}
    domain_pref = domain_df[domain_df["domain_source"].isin(source_rank)].copy()

    if domain_pref.empty:
        sample_df["domain_count"] = np.nan
        sample_df["domain_group"] = "unknown"
        sample_df["domain_source_used"] = "none"
        sample_df = sample_df.drop(columns=["_pdb_id_norm", "_chain_id_norm"])
        return sample_df

    domain_pref["_source_rank"] = domain_pref["domain_source"].map(source_rank)

    domain_counts = (
        domain_pref
        .groupby(["pdb_id", "chain_id", "domain_source", "_source_rank"], as_index=False)["domain_id"]
        .nunique()
        .rename(columns={"domain_id": "domain_count"})
    )

    best_source = (
        domain_counts
        .sort_values(["pdb_id", "chain_id", "_source_rank"])
        .drop_duplicates(["pdb_id", "chain_id"], keep="first")
    )

    best_source["domain_source_used"] = best_source["domain_source"]
    best_source["domain_group"] = np.select(
        [
            best_source["domain_count"] == 1,
            best_source["domain_count"] > 1,
        ],
        [
            "single_domain",
            "multi_domain",
        ],
        default="unknown",
    )

    sample_df = sample_df.merge(
        best_source[
            [
                "pdb_id",
                "chain_id",
                "domain_count",
                "domain_group",
                "domain_source_used",
            ]
        ],
        left_on=["_pdb_id_norm", "_chain_id_norm"],
        right_on=["pdb_id", "chain_id"],
        how="left",
    )

    sample_df["domain_count"] = pd.to_numeric(sample_df["domain_count"], errors="coerce")
    sample_df["domain_group"] = sample_df["domain_group"].fillna("unknown")
    sample_df["domain_source_used"] = sample_df["domain_source_used"].fillna("none")

    sample_df = sample_df.drop(columns=["_pdb_id_norm", "_chain_id_norm", "pdb_id", "chain_id"])
    return sample_df

def _load_resolution_cache(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_resolution_cache(path: Path, cache: dict[str, float | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2)


def _parse_local_resolution(pdb_path: Path) -> float | None:
    with gzip.open(pdb_path, "rt", encoding="utf-8", errors="replace") as handle:
        mmcif = MMCIF2Dict(handle)
    for key in ["_refine.ls_d_res_high", "_em_3d_reconstruction.resolution"]:
        value = mmcif.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            value = value[0]
        try:
            return float(value)
        except Exception:
            continue
    return None


def _fetch_resolution_from_rcsb(pdb_id: str) -> float | None:
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
    request = urllib.request.Request(url, headers={"User-Agent": "MuSRNet/1.0"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    values = payload.get("rcsb_entry_info", {}).get("resolution_combined")
    if values:
        try:
            return float(values[0])
        except Exception:
            return None
    return None


# def load_or_fetch_resolution(sample_row: pd.Series, pdb_dir: Path, cache_path: Path, warnings: list[str]) -> float:
#     for key in ["resolution", "wt_resolution"]:
#         if key in sample_row and pd.notna(sample_row[key]):
#             return float(sample_row[key])

#     pdb_id = str(sample_row["wt_pdb_id"]).lower()
#     cache = _load_resolution_cache(cache_path)
#     if pdb_id in cache:
#         value = cache[pdb_id]
#         if value is None:
#             _warn(warnings, f"resolution unknown for {pdb_id}; cached missing value")
#             return float("nan")
#         return float(value)

#     try:
#         pdb_path = find_pdb_file(str(pdb_dir), pdb_id)
#         resolution = _parse_local_resolution(pdb_path)
#         if resolution is not None:
#             cache[pdb_id] = float(resolution)
#             _save_resolution_cache(cache_path, cache)
#             return float(resolution)
#     except Exception as exc:
#         _warn(warnings, f"local resolution unavailable for {pdb_id}: {exc}")

#     try:
#         resolution = _fetch_resolution_from_rcsb(pdb_id)
#     except Exception as exc:
#         cache[pdb_id] = None
#         _save_resolution_cache(cache_path, cache)
#         _warn(warnings, f"RCSB resolution query failed for {pdb_id}: {exc}")
#         return float("nan")

#     cache[pdb_id] = None if resolution is None else float(resolution)
#     _save_resolution_cache(cache_path, cache)
#     if resolution is None:
#         _warn(warnings, f"resolution unknown for {pdb_id}; RCSB returned no value")
#         return float("nan")
#     return float(resolution)

def load_or_fetch_resolution(sample_row: pd.Series, pdb_dir: Path, cache: dict, warnings: list[str]) -> float:
    for key in ["resolution", "wt_resolution"]:
        if key in sample_row and pd.notna(sample_row[key]):
            return float(sample_row[key])

    pdb_id = str(sample_row["wt_pdb_id"]).lower()
    
    if pdb_id in cache:
        value = cache[pdb_id]
        if value is None:
            _warn(warnings, f"resolution unknown for {pdb_id}; cached missing value")
            return float("nan")
        return float(value)

    try:
        pdb_path = find_pdb_file(str(pdb_dir), pdb_id)
        resolution = _parse_local_resolution(pdb_path)
        if resolution is not None:
            cache[pdb_id] = float(resolution)
            return float(resolution)
    except Exception as exc:
        _warn(warnings, f"local resolution unavailable for {pdb_id}: {exc}")

    try:
        resolution = _fetch_resolution_from_rcsb(pdb_id)
    except Exception as exc:
        cache[pdb_id] = None
        _warn(warnings, f"RCSB resolution query failed for {pdb_id}: {exc}")
        return float("nan")

    cache[pdb_id] = None if resolution is None else float(resolution)
    if resolution is None:
        _warn(warnings, f"resolution unknown for {pdb_id}; RCSB returned no value")
        return float("nan")
    return float(resolution)

def _decompress_to_temp(cif_gz_path: Path) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmpdir = tempfile.TemporaryDirectory(prefix="musrnet_dssp_")
    tmp_path = Path(tmpdir.name) / cif_gz_path.with_suffix("").name
    with gzip.open(cif_gz_path, "rt", encoding="utf-8", errors="replace") as src, tmp_path.open("w", encoding="utf-8") as dst:
        dst.write(src.read())
    return tmpdir, tmp_path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _compute_freesasa_annotation(structure, chain_id: str, label_chain_ids: list[str]) -> dict[tuple[int, str], dict[str, float]]:
    result, classes = freesasa.calcBioPDB(structure)
    residue_areas = result.residueAreas()
    chain_areas = residue_areas.get(chain_id, {})
    if not chain_areas:
        for label_chain_id in label_chain_ids:
            chain_areas = residue_areas.get(label_chain_id, {})
            if chain_areas:
                break
    mapping: dict[tuple[int, str], dict[str, float]] = {}
    for residue_label, area in chain_areas.items():
        digits = "".join(ch for ch in residue_label if ch.isdigit() or ch == "-")
        icode = residue_label[len(digits) :].strip()
        try:
            resseq = int(digits)
        except Exception:
            continue
        resname = classes[chain_id][residue_label].residueType if chain_id in classes and residue_label in classes[chain_id] else None
        max_asa = MAX_ASA.get(resname, np.nan) if resname else np.nan
        rsa = float(area.total / max_asa) if max_asa and np.isfinite(max_asa) else float("nan")
        mapping[(resseq, icode)] = {"rsa": float(np.clip(rsa, 0.0, 1.0)) if np.isfinite(rsa) else float("nan")}
    return mapping

def _load_downloaded_dssp_chain_cache(dssp_path: Path) -> dict[str, dict[str, Any]]:
    dssp_dict, _ = make_dssp_dict(str(dssp_path))
    max_acc = residue_max_acc["Sander"]
    chain_cache: dict[str, dict[str, Any]] = {}
    for key, values in dssp_dict.items():
        chain = key[0]
        residue_id = key[1]

        resseq = int(residue_id[1])
        icode = str(residue_id[2]).strip()

        aa = values[0]
        ss = values[1]
        acc = values[2]

        aa_for_scale = 'C' if str(aa).islower() else str(aa).upper()
        resname = protein_letters_1to3.get(aa_for_scale, None)
        
        try:
            rsa = float(acc) / float(max_acc[resname])
            rsa = min(rsa, 1.0)
        except Exception:
            rsa = float("nan")

        chain_cache.setdefault(chain, {})
        chain_cache[chain][str(resseq)] = {
            "rsa": rsa,
            "secondary_structure_code": ss if ss else "unknown",
            "annotation_status": f"dssp_download:{dssp_path.name}",
            "icode": icode,
        }
    return chain_cache

def _as_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x]
    return [str(x).strip()]


def _chain_aliases_from_mmcif(cif_path: Path, chain_id: str) -> list[str]:
    """
    Build safe chain aliases from mmCIF:
        label_asym_id <-> auth_asym_id

    Example:
        label_asym_id = 'AA'
        auth_asym_id  = 'A'
        chain_id 'AA' can safely match DSSP chain 'A'.
    """
    chain_id = str(chain_id).strip()
    aliases = [chain_id]

    try:
        mmcif = MMCIF2Dict(str(cif_path))
    except Exception:
        return aliases

    labels = _as_list(mmcif.get("_atom_site.label_asym_id"))
    auths = _as_list(mmcif.get("_atom_site.auth_asym_id"))

    for label, auth in zip(labels, auths):
        if not label or not auth:
            continue
        if label in {".", "?"} or auth in {".", "?"}:
            continue

        if label == chain_id and auth not in aliases:
            aliases.append(auth)

        if auth == chain_id and label not in aliases:
            aliases.append(label)

    return aliases

def _run_dssp_single_chain_legacy_safe(structure, source_chain_id: str):
    """
    Full structure may not fit legacy DSSP.
    Workaround: keep only target chain, rename it to 'A', then run DSSP.
    """
    single_structure = copy.deepcopy(structure)

    model_ids = [model.id for model in single_structure]
    first_model_id = model_ids[0]

    for model_id in model_ids[1:]:
        single_structure.detach_child(model_id)

    model = single_structure[first_model_id]

    if source_chain_id not in model.child_dict:
        raise ValueError(f"chain_not_found_for_single_chain_dssp: {source_chain_id}")

    for cid in list(model.child_dict.keys()):
        if cid != source_chain_id:
            model.detach_child(cid)

    chain = model[source_chain_id]
    chain.id = "A"

    with tempfile.TemporaryDirectory() as td:
        chain_cif = Path(td) / "single_chain_for_dssp.cif"

        io = MMCIFIO()
        io.set_structure(single_structure)
        io.save(str(chain_cif))

        chain_model = next(single_structure.get_models())

        return DSSP(chain_model, str(chain_cif), dssp="mkdssp")

def load_or_compute_dssp_annotation(
    pdb_id: str,
    chain_id: str,
    residue_number: int,
    dssp_cache_dir: Path,
    pdb_dir: Path,
    warnings: list[str],
    dssp_download_dir: Path = Path("data/dssp_downloads"),
) -> dict[str, Any]:
    cache_path = dssp_cache_dir / f"{pdb_id.lower()}_{chain_id}.json"
    cache = _load_json(cache_path)
    cache_key = str(int(residue_number))
    if cache_key in cache:
        return cache[cache_key]

    downloaded_dssp_checked = False
    downloaded_dssp_path = dssp_download_dir / f"{pdb_id}.dssp"

    if downloaded_dssp_path is not None:
        try:
            downloaded_chain_cache = _load_downloaded_dssp_chain_cache(downloaded_dssp_path)
            downloaded_dssp_checked = True

            if chain_id in downloaded_chain_cache:
                _save_json(cache_path, downloaded_chain_cache[chain_id])

                if cache_key in downloaded_chain_cache[chain_id]:
                    return downloaded_chain_cache[chain_id][cache_key]
            
            _warn(
                warnings,
                f"{pdb_id}:{chain_id}:{residue_number} downloaded_DSSP_missing_residue: {downloaded_dssp_path}",
            )
        except Exception as exc:
            _warn(
                warnings,
                f"{pdb_id}:{chain_id}:{residue_number} downloaded_DSSP_unusable: {downloaded_dssp_path}: {exc}",
            )
    try:
        pdb_path = find_pdb_file(str(pdb_dir), pdb_id)
    except Exception as exc:
        _warn(warnings, f"{pdb_id}:{chain_id}:{residue_number} missing_structure_file: {exc}")
        return {"rsa": float("nan"), "secondary_structure_code": "unknown", "annotation_status": "missing_structure_file"}
    tmpdir, tmp_path = _decompress_to_temp(pdb_path)
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(tmp_path))
        model = next(structure.get_models())
        label_chain_id = None
        label_chain_ids = _chain_aliases_from_mmcif(tmp_path, chain_id)

        if downloaded_dssp_checked and downloaded_chain_cache is not None:
            for alias in label_chain_ids:
                if alias in downloaded_chain_cache:
                    label_chain_id = alias
                    break
            if label_chain_id is not None:
                _save_json(cache_path, downloaded_chain_cache[label_chain_id])
                if cache_key in downloaded_chain_cache[label_chain_id]:
                    result = downloaded_chain_cache[label_chain_id][cache_key]
                    result = dict(result)
                    result['annotation_status'] = (
                        f'{result.get("annotation_status", "downloaded_dssp")}'
                        f":chain_id_mapped:{chain_id}->{label_chain_id}"
                    )
                    return result
                _warn(
                    warnings,
                    f"{pdb_id}:{chain_id}:{residue_number} downloaded_DSSP_missing_residue_or_chain "
                    f"available_chains={sorted(downloaded_chain_cache.keys())}: {downloaded_dssp_path}",
                )

        dssp_result = None
        dssp_reason = None

        if not downloaded_dssp_checked:
            try:
                dssp_result = DSSP(model, str(tmp_path), dssp='mkdssp')
                dssp_reason = 'mkdssp'
            except Exception as exc:
                dssp_reason = str(exc)
        else:
            dssp_reason = f"downloaded_DSSP_checked:{downloaded_dssp_path}"

        if dssp_result is not None:
            chain_cache = {}
            for key in dssp_result.keys():
                chain = key[0]
                resseq = int(key[1][1])
                icode = str(key[1][2]).strip()
                values = dssp_result[key]

                rsa_value = values[3]
                try:
                    rsa = float(rsa_value)
                except (TypeError, ValueError):
                    rsa = float("nan")

                chain_cache.setdefault(chain, {})
                chain_cache[chain][str(resseq)] = {
                    "rsa": rsa,
                    "secondary_structure_code": values[2] if values[2] else "unknown",
                    "annotation_status": f"dssp:{dssp_reason}",
                    "icode": icode,
                }
            if chain_id in chain_cache:
                _save_json(cache_path, chain_cache[chain_id]) 
                if cache_key in chain_cache[chain_id]:
                    return chain_cache[chain_id][cache_key]
            else:
                for label_chain_id in label_chain_ids:
                    if label_chain_id in chain_cache:
                        _save_json(cache_path, chain_cache[label_chain_id])
                        if cache_key in chain_cache[label_chain_id]:
                            return chain_cache[label_chain_id][cache_key]
        try:
            freesasa_cache = _compute_freesasa_annotation(structure, chain_id, label_chain_ids)
            result = freesasa_cache.get((int(residue_number), ""))
            if result is not None:
                response = {
                    "rsa": result["rsa"],
                    "secondary_structure_code": "unknown",
                    "annotation_status": "freesasa_rsa_only",
                }
                cache[cache_key] = response
                _save_json(cache_path, cache)
                _warn(warnings, f"{pdb_id}:{chain_id}:{residue_number} DSSP unavailable; used FreeSASA fallback")
                return response
        except Exception as exc:
            _warn(warnings, f"{pdb_id}:{chain_id}:{residue_number} FreeSASA unavailable: {exc}")

        _warn(warnings, f"{pdb_id}:{chain_id}:{residue_number} annotation unknown; DSSP unavailable: {dssp_reason}")
        return {"rsa": float("nan"), "secondary_structure_code": "unknown", "annotation_status": f"unknown:{dssp_reason}"}
    finally:
        tmpdir.cleanup()

# def load_or_compute_dssp_annotation(
#     pdb_id: str,
#     chain_id: str,
#     residue_number: int,
#     dssp_cache_dir: Path,
#     pdb_dir: Path,
#     warnings: list[str],
# ) -> dict[str, Any]:
#     cache_path = dssp_cache_dir / f"{pdb_id.lower()}_{chain_id}.json"
#     cache = _load_json(cache_path)
#     cache_key = str(int(residue_number))
#     if cache_key in cache:
#         return cache[cache_key]

#     try:
#         pdb_path = find_pdb_file(str(pdb_dir), pdb_id)
#     except Exception as exc:
#         _warn(warnings, f"{pdb_id}:{chain_id}:{residue_number} missing_structure_file: {exc}")
#         return {"rsa": float("nan"), "secondary_structure_code": "unknown", "annotation_status": "missing_structure_file"}

#     tmpdir, tmp_path = _decompress_to_temp(pdb_path)
#     try:
#         parser = MMCIFParser(QUIET=True)
#         structure = parser.get_structure(pdb_id, str(tmp_path))
#         model = next(structure.get_models())

#         dssp_result = None
#         dssp_reason = None
#         for exe_name in ["mkdssp", "dssp"]:
#             try:
#                 from Bio.PDB.DSSP import DSSP

#                 dssp_result = DSSP(model, str(tmp_path), dssp=exe_name)
#                 dssp_reason = exe_name
#                 break
#             except Exception as exc:
#                 dssp_reason = str(exc)

#         if dssp_result is not None:
#             chain_cache = {}
#             for key in dssp_result.keys():
#                 chain = key[0]
#                 resseq = int(key[1][1])
#                 icode = str(key[1][2]).strip()
#                 values = dssp_result[key]

#                 rsa_value = values[3]
#                 try:
#                     rsa = float(rsa_value)
#                 except (TypeError, ValueError):
#                     rsa = float("nan")

#                 chain_cache.setdefault(chain, {})
#                 chain_cache[chain][str(resseq)] = {
#                     "rsa": rsa,
#                     "secondary_structure_code": values[2] if values[2] else "unknown",
#                     "annotation_status": f"dssp:{dssp_reason}",
#                     "icode": icode,
#                 }
#             if chain_id in chain_cache:
#                 _save_json(cache_path, chain_cache[chain_id])
#                 if cache_key in chain_cache[chain_id]:
#                     return chain_cache[chain_id][cache_key]

#         try:
#             freesasa_cache = _compute_freesasa_annotation(structure, chain_id)
#             result = freesasa_cache.get((int(residue_number), ""))
#             if result is not None:
#                 response = {
#                     "rsa": result["rsa"],
#                     "secondary_structure_code": "unknown",
#                     "annotation_status": "freesasa_rsa_only",
#                 }
#                 cache[cache_key] = response
#                 _save_json(cache_path, cache)
#                 _warn(warnings, f"{pdb_id}:{chain_id}:{residue_number} DSSP unavailable; used FreeSASA fallback")
#                 return response
#         except Exception as exc:
#             _warn(warnings, f"{pdb_id}:{chain_id}:{residue_number} FreeSASA unavailable: {exc}")

#         _warn(warnings, f"{pdb_id}:{chain_id}:{residue_number} annotation unknown; DSSP unavailable: {dssp_reason}")
#         return {"rsa": float("nan"), "secondary_structure_code": "unknown", "annotation_status": f"unknown:{dssp_reason}"}
#     finally:
#         tmpdir.cleanup()

def build_biological_annotations(
    sample_df: pd.DataFrame,
    pdb_dir: str | Path,
    args,
    num_workers: int = 1,
) -> pd.DataFrame:
    warnings: list[str] = args.warnings
    cache_path = Path(args.resolution_cache)
    dssp_cache_dir = Path(args.dssp_cache_dir)
    pdb_dir = Path(pdb_dir)

    df = sample_df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["cluster_id_30"] = df["cluster_id_30"].astype(str)
    df["wt_pdb_id"] = df["wt_pdb_id"].astype(str).str.lower()
    df["wt_chain_id"] = df["wt_chain_id"].astype(str)
    df["mut_pos_pdb_number"] = pd.to_numeric(df["mut_pos_pdb_number"], errors="coerce")

    # ---- domain annotations ----
    domain_df = load_domain_annotations(getattr(args, "domain_annotations", None), warnings)
    df = assign_domain_group_for_samples(
        sample_df=df,
        domain_df=domain_df,
        preferred_sources=("CATH", "SCOP", "Pfam", "InterPro"),
        warnings=warnings,
    )

    # ---- basic biological groups ----
    df["wt_aa"] = df["wt_aa_type"].astype(str)
    df["mut_aa"] = df["mut_aa_type"].astype(str)

    df["length"] = df["wt_sequence"].astype(str).str.len()
    df["length_group"] = df["length"].apply(
        lambda x: assign_length_group(int(x), float(args.long_protein_threshold_value))
    )

    df["charge_binary_group"] = np.where(
        df["wt_aa"].isin(CHARGED) | df["mut_aa"].isin(CHARGED),
        "charge_involved",
        "no_charge_involved",
    )

    df["gly_group"] = np.where(
        (df["wt_aa"] == "G") | (df["mut_aa"] == "G"),
        "gly_involved",
        "no_gly",
    )

    df["pro_group"] = np.where(
        (df["wt_aa"] == "P") | (df["mut_aa"] == "P"),
        "pro_involved",
        "no_pro",
    )

    df["substitution_size_group"] = df.apply(
        lambda r: assign_substitution_size_group(r["wt_aa"], r["mut_aa"]),
        axis=1,
    )

    df["charge_group"] = df.apply(
        lambda r: assign_charge_group(r["wt_aa"], r["mut_aa"]),
        axis=1,
    )

    df["glypro_group"] = df.apply(
        lambda r: assign_glypro_group(r["wt_aa"], r["mut_aa"]),
        axis=1,
    )

    # ---- resolution + DSSP/RSA annotations ----
    resolution_cache_dict = _load_resolution_cache(cache_path)

    def fetch_external_data(row_dict: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        row_series = pd.Series(row_dict)

        resolution = load_or_fetch_resolution(
            sample_row=row_series,
            pdb_dir=pdb_dir,
            cache=resolution_cache_dict,
            warnings=warnings,
        )

        if pd.isna(row_dict.get("mut_pos_pdb_number")):
            _warn(
                warnings,
                f"{row_dict['sample_id']} missing mut_pos_pdb_number; DSSP/RSA annotation skipped",
            )
            dssp = {
                "rsa": float("nan"),
                "secondary_structure_code": "unknown",
                "annotation_status": "missing_mut_pos_pdb_number",
            }
            return resolution, dssp

        dssp = load_or_compute_dssp_annotation(
            pdb_id=str(row_dict["wt_pdb_id"]).lower(),
            chain_id=str(row_dict["wt_chain_id"]),
            residue_number=int(row_dict["mut_pos_pdb_number"]),
            dssp_cache_dir=dssp_cache_dir,
            pdb_dir=pdb_dir,
            warnings=warnings,
            dssp_download_dir=Path(args.dssp_download_dir),
        )

        return resolution, dssp

    records = df.to_dict(orient="records")

    if num_workers <= 1:
        results = [
            fetch_external_data(record)
            for record in tqdm(records, total=len(records), desc="Fetching DSSP/Resolution")
        ]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(fetch_external_data, records),
                    total=len(records),
                    desc="Fetching DSSP/Resolution",
                )
            )

    _save_resolution_cache(cache_path, resolution_cache_dict)

    df["resolution"] = [r[0] for r in results]
    dssp_infos = [r[1] for r in results]

    df["rsa_mut_site"] = [d.get("rsa", float("nan")) for d in dssp_infos]
    df["exposure_group"] = df["rsa_mut_site"].apply(
        lambda x: assign_exposure_group(x, float(args.rsa_threshold))
    )

    df["secondary_structure_code"] = [
        d.get("secondary_structure_code", "unknown") for d in dssp_infos
    ]

    df["secondary_structure_group"] = df["secondary_structure_code"].apply(
        assign_secondary_structure_group
    )

    df["annotation_status"] = [
        d.get("annotation_status", "unknown") for d in dssp_infos
    ]

    df["resolution_group"] = df["resolution"].apply(
        lambda x: assign_resolution_group(x, float(args.high_resolution_threshold))
    )

    df = df.drop(columns=["wt_aa", "mut_aa"], errors="ignore")

    return df

# def build_biological_annotations(sample_df: pd.DataFrame, pdb_dir: str | Path, args) -> pd.DataFrame:
#     warnings: list[str] = args.warnings
#     domain_df = load_domain_annotations(getattr(args, "domain_annotations", None), warnings)
#     cache_path = Path(args.resolution_cache)
#     dssp_cache_dir = Path(args.dssp_cache_dir)
#     pdb_dir = Path(pdb_dir)

#     rows = []
#     for sample in sample_df.to_dict(orient="records"):
#         row = dict(sample)
#         wt_aa = str(row["wt_aa_type"])
#         mut_aa = str(row["mut_aa_type"])
#         length = int(len(str(row["wt_sequence"])))
#         resolution = load_or_fetch_resolution(pd.Series(row), pdb_dir, cache_path, warnings)
#         dssp_info = load_or_compute_dssp_annotation(
#             pdb_id=str(row["wt_pdb_id"]).lower(),
#             chain_id=str(row["wt_chain_id"]),
#             residue_number=int(row["mut_pos_pdb_number"]),
#             dssp_cache_dir=dssp_cache_dir,
#             pdb_dir=pdb_dir,
#             warnings=warnings,
#         )
#         domain_rows = domain_df[
#             (domain_df["pdb_id"] == str(row["wt_pdb_id"]).lower()) & (domain_df["chain_id"] == str(row["wt_chain_id"]))
#         ]
#         domain_count = float(domain_rows["domain_id"].nunique()) if not domain_rows.empty else float("nan")
#         domain_group = "unknown" if domain_rows.empty else ("single_domain" if int(domain_count) == 1 else "multi_domain")
#         annotation_status = dssp_info.get("annotation_status", "ok")

#         row["length"] = length
#         row["length_group"] = assign_length_group(length, float(args.long_protein_threshold_value))
#         row["substitution_size_group"] = assign_substitution_size_group(wt_aa, mut_aa)
#         row["charge_group"] = assign_charge_group(wt_aa, mut_aa)
#         row["charge_binary_group"] = "charge_involved" if wt_aa in CHARGED or mut_aa in CHARGED else "no_charge_involved"
#         row["glypro_group"] = assign_glypro_group(wt_aa, mut_aa)
#         row["gly_group"] = "gly_involved" if wt_aa == "G" or mut_aa == "G" else "no_gly"
#         row["pro_group"] = "pro_involved" if wt_aa == "P" or mut_aa == "P" else "no_pro"
#         row["rsa_mut_site"] = dssp_info.get("rsa", float("nan"))
#         row["exposure_group"] = assign_exposure_group(row["rsa_mut_site"], float(args.rsa_threshold))
#         row["secondary_structure_code"] = dssp_info.get("secondary_structure_code", "unknown")
#         row["secondary_structure_group"] = assign_secondary_structure_group(row["secondary_structure_code"])
#         row["resolution"] = resolution
#         row["resolution_group"] = assign_resolution_group(resolution, float(args.high_resolution_threshold))
#         row["domain_count"] = domain_count
#         row["domain_group"] = domain_group
#         row["annotation_status"] = annotation_status
#         rows.append(row)
#     return pd.DataFrame(rows)


def _derive_radius_and_class(
    sample_df: pd.DataFrame,
    response_threshold: float,
    displacement_threshold: float,
    radius_threshold: float,
    warnings: list[str],
) -> dict[str, float]:
    if "radii" not in sample_df.columns or sample_df["radii"].isna().all():
        _warn(warnings, f"radii unavailable for sample {sample_df['sample_id'].iloc[0]}; radius/class metrics skipped")
        return {
            "true_radius": float("nan"),
            "pred_radius": float("nan"),
            "true_class": float("nan"),
            "pred_class": float("nan"),
            "radius_mae": float("nan"),
            "class_correct": float("nan"),
            "class_macro_f1": float("nan"),
            "non_local_f1": float("nan"),
            "non_local_recall": float("nan"),
            "non_local_class_status": "not_assessed",
        }

    radii = sample_df["radii"].to_numpy(dtype=np.float64)
    pred_disp = sample_df["pred_displacement"].to_numpy(dtype=np.float64)
    pred_prob = sample_df["pred_perturbed_prob"].to_numpy(dtype=np.float64)
    true_disp = sample_df["true_displacement"].to_numpy(dtype=np.float64)

    pred_score = pred_disp * pred_prob
    pred_mask = pred_score > response_threshold
    pred_radius = float(radii[pred_mask].max()) if pred_mask.any() else 0.0

    if "true_radius" in sample_df.columns and sample_df["true_radius"].notna().any():
        true_radius = float(sample_df["true_radius"].iloc[0])
    else:
        true_mask = true_disp > displacement_threshold
        true_radius = float(radii[true_mask].max()) if true_mask.any() else 0.0

    if "true_class" in sample_df.columns and sample_df["true_class"].notna().any():
        true_class = int(sample_df["true_class"].iloc[0])
    else:
        max_true = float(true_disp.max()) if true_disp.size else 0.0
        true_class = 0 if max_true <= displacement_threshold else (1 if true_radius <= radius_threshold else 2)

    if "pred_class" in sample_df.columns and sample_df["pred_class"].notna().any():
        pred_class = int(sample_df["pred_class"].iloc[0])
    else:
        max_pred = float(pred_disp.max()) if pred_disp.size else 0.0
        pred_class = 0 if max_pred <= displacement_threshold else (1 if pred_radius <= radius_threshold else 2)

    true_non_local = np.array([1 if true_class == 2 else 0], dtype=np.int64)
    pred_non_local = np.array([1 if pred_class == 2 else 0], dtype=np.int64)
    return {
        "true_radius": true_radius,
        "pred_radius": pred_radius,
        "true_class": true_class,
        "pred_class": pred_class,
        "radius_mae": abs(pred_radius - true_radius),
        "class_correct": float(pred_class == true_class),
        "class_macro_f1": float(f1_score([true_class], [pred_class], average="macro", labels=[0, 1, 2], zero_division=0)),
        "non_local_f1": float(f1_score(true_non_local, pred_non_local, zero_division=0)),
        "non_local_recall": float(pred_non_local[0] == 1) if true_non_local[0] == 1 else float("nan"),
        "non_local_class_status": "derived",
    }


def compute_sample_metrics(
    df: pd.DataFrame,
    response_threshold: float = 0.5,
    displacement_threshold: float = 1.0,
    radius_threshold: float = 8.0,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    warnings = warnings if warnings is not None else []
    rows = []
    for sample_id, sample_df in df.groupby("sample_id", sort=False):
        errors = np.abs(
            sample_df["pred_displacement"].to_numpy(dtype=np.float64)
            - sample_df["true_displacement"].to_numpy(dtype=np.float64)
        )
        row = {
            "sample_id": str(sample_id),
            "cluster_id_30": str(sample_df["cluster_id_30"].iloc[0]),
            "global_mae": float(errors.mean()) if errors.size else float("nan"),
            "n_residues": int(len(sample_df)),
        }
        shell_values = []
        for shell_idx in range(5):
            mask = sample_df["shell_id"].to_numpy(dtype=np.int64) == shell_idx
            key = f"mae_shell_{shell_idx}"
            row[key] = float(errors[mask].mean()) if mask.any() else float("nan")
            if mask.any():
                shell_values.append(row[key])
        row["shell_mae"] = float(np.mean(shell_values)) if shell_values else float("nan")

        y_true = sample_df["true_perturbed"].to_numpy(dtype=np.int64)
        y_score = sample_df["pred_perturbed_prob"].to_numpy(dtype=np.float64)
        if np.unique(y_true).size < 2:
            row["perturbed_auroc"] = float("nan")
            row["perturbed_auprc"] = float("nan")
            _warn(warnings, f"AUROC/AUPRC undefined for one-class sample {sample_id}")
        else:
            row["perturbed_auroc"] = float(roc_auc_score(y_true, y_score))
            row["perturbed_auprc"] = float(average_precision_score(y_true, y_score))

        row.update(
            _derive_radius_and_class(
                sample_df=sample_df,
                response_threshold=response_threshold,
                displacement_threshold=displacement_threshold,
                radius_threshold=radius_threshold,
                warnings=warnings,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _cluster_bootstrap(values: pd.DataFrame, metric: str, bootstrap_iters: int, seed: int) -> tuple[float, float]:
    arr = values[metric].to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 1:
        return float("nan"), float("nan")
    return paired_bootstrap_ci(arr, n_bootstrap=bootstrap_iters, seed=seed)


def compute_stratified_metrics(
    sample_metrics_df: pd.DataFrame,
    factor_columns: list[str],
    metric_columns: list[str],
    bootstrap_iters: int,
    seed: int,
    warnings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []

    for model_name, model_df in sample_metrics_df.groupby("model_name", sort=False):
        for factor in factor_columns:
            if factor not in model_df.columns:
                continue
            for group, group_df in model_df.groupby(factor, dropna=False, sort=False):
                for metric in metric_columns:
                    if metric not in group_df.columns:
                        continue
                    valid_df = group_df[group_df[metric].notna()].copy()
                    cluster_df = (
                        valid_df.groupby("cluster_id_30", dropna=False)
                        .agg(
                            metric_value=(metric, "mean"),
                            n_samples=("sample_id", "nunique"),
                            n_residues=("n_residues", "sum"),
                        )
                        .reset_index()
                    )
                    if len(cluster_df) < 5:
                        _warn(warnings, f"fewer than 5 clusters in stratum {factor}={group} for {model_name}/{metric}")
                    ci_low, ci_high = _cluster_bootstrap(cluster_df.rename(columns={"metric_value": metric}), metric, bootstrap_iters, seed) if len(cluster_df) >= 5 else (float("nan"), float("nan"))
                    long_rows.append(
                        {
                            "model_name": model_name,
                            "factor": factor,
                            "group": group,
                            "metric": metric,
                            "sample_mean": float(valid_df[metric].mean()) if not valid_df.empty else float("nan"),
                            "cluster_mean": float(cluster_df["metric_value"].mean()) if not cluster_df.empty else float("nan"),
                            "sample_weighted_cluster_mean": float(np.average(cluster_df["metric_value"], weights=cluster_df["n_samples"])) if not cluster_df.empty else float("nan"),
                            "residue_weighted_cluster_mean": float(np.average(cluster_df["metric_value"], weights=cluster_df["n_residues"])) if not cluster_df.empty else float("nan"),
                            "n_samples": int(group_df["sample_id"].nunique()),
                            "n_clusters": int(group_df["cluster_id_30"].nunique()),
                            "n_residues": int(group_df["n_residues"].sum()),
                            "n_valid_samples": int(valid_df["sample_id"].nunique()),
                            "bootstrap_ci_low": ci_low,
                            "bootstrap_ci_high": ci_high,
                        }
                    )
                    for cluster_row in cluster_df.to_dict(orient="records"):
                        cluster_rows.append(
                            {
                                "model_name": model_name,
                                "factor": factor,
                                "group": group,
                                "metric": metric,
                                "cluster_id_30": str(cluster_row["cluster_id_30"]),
                                "value": cluster_row["metric_value"],
                                "n_samples": int(cluster_row["n_samples"]),
                                "n_residues": int(cluster_row["n_residues"]),
                            }
                        )
    return pd.DataFrame(long_rows), pd.DataFrame(cluster_rows)


def compute_pairwise_stratified_diffs(
    cluster_metric_df: pd.DataFrame,
    reference: str,
    candidate: str,
    bootstrap_iters: int,
    seed: int,
    warnings: list[str],
) -> pd.DataFrame:
    rows = []
    grouped = cluster_metric_df.groupby(["factor", "group", "metric"], dropna=False, sort=False)
    for (factor, group, metric), group_df in grouped:
        ref_df = group_df[group_df["model_name"] == reference][["cluster_id_30", "value", "n_samples", "n_residues"]]
        cand_df = group_df[group_df["model_name"] == candidate][["cluster_id_30", "value", "n_samples", "n_residues"]]
        merged = ref_df.merge(cand_df, on="cluster_id_30", how="inner", suffixes=("_reference", "_candidate"))
        diff = merged["value_candidate"].to_numpy(dtype=np.float64) - merged["value_reference"].to_numpy(dtype=np.float64)
        valid = np.isfinite(diff)
        diff = diff[valid]
        if diff.size < 5:
            _warn(warnings, f"fewer than 5 common valid clusters for pairwise stratified diff {factor}={group}/{metric}")
            ci_low, ci_high, pvalue = float("nan"), float("nan"), float("nan")
        else:
            ci_low, ci_high = paired_bootstrap_ci(diff, n_bootstrap=bootstrap_iters, seed=seed)
            pvalue = wilcoxon_signed_rank_pvalue(diff)
        higher_is_better = metric in HIGHER_IS_BETTER_METRICS
        improved = diff > 0 if higher_is_better else diff < 0
        worsened = diff < 0 if higher_is_better else diff > 0
        weights_samples = merged.loc[valid, "n_samples_candidate"].to_numpy(dtype=np.float64) if valid.any() else np.array([], dtype=np.float64)
        weights_residues = merged.loc[valid, "n_residues_candidate"].to_numpy(dtype=np.float64) if valid.any() else np.array([], dtype=np.float64)
        rows.append(
            {
                "factor": factor,
                "group": group,
                "metric": metric,
                "reference": reference,
                "candidate": candidate,
                "n_common_clusters": int(diff.size),
                "mean_diff": float(np.mean(diff)) if diff.size else float("nan"),
                "median_diff": float(np.median(diff)) if diff.size else float("nan"),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "wilcoxon_p": pvalue,
                "n_clusters_improved": int(np.sum(improved)),
                "n_clusters_worsened": int(np.sum(worsened)),
                "fraction_clusters_improved": float(np.mean(improved)) if diff.size else float("nan"),
                "weighted_mean_diff_by_n_samples": float(np.average(diff, weights=weights_samples)) if diff.size and weights_samples.sum() else float("nan"),
                "weighted_mean_diff_by_n_residues": float(np.average(diff, weights=weights_residues)) if diff.size and weights_residues.sum() else float("nan"),
            }
        )
    return pd.DataFrame(rows)
