from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    coords_wt_aligned: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    mask: np.ndarray
    alignment_name: str
    alignment_rmsd: float
    n_aligned_residues: int
    metadata: dict


def _validate_inputs(coords_wt: np.ndarray, coords_mut: np.ndarray, mask: np.ndarray | None = None) -> None:
    if coords_wt.shape != coords_mut.shape:
        raise ValueError("coords_wt and coords_mut must have the same shape")
    if coords_wt.ndim != 2 or coords_wt.shape[1] != 3:
        raise ValueError("Coordinates must have shape [L, 3]")
    if not np.isfinite(coords_wt).all() or not np.isfinite(coords_mut).all():
        raise ValueError("Coordinates contain NaN or Inf")
    if mask is not None and mask.shape != (coords_wt.shape[0],):
        raise ValueError("Mask must have shape [L]")


def kabsch_transform(
    coords_wt: np.ndarray,
    coords_mut: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    _validate_inputs(coords_wt, coords_mut, mask)
    mask = mask.astype(bool, copy=False)
    if mask.sum() < 3:
        raise ValueError("Kabsch alignment requires at least 3 residues in mask")

    X = coords_wt[mask]
    Y = coords_mut[mask]

    x_centroid = X.mean(axis=0)
    y_centroid = Y.mean(axis=0)

    X0 = X - x_centroid
    Y0 = Y - y_centroid

    H = X0.T @ Y0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = y_centroid - x_centroid @ R.T
    coords_wt_aligned = coords_wt @ R.T + t
    rmsd = float(np.sqrt(np.mean(np.sum((coords_wt_aligned[mask] - coords_mut[mask]) ** 2, axis=1))))
    return coords_wt_aligned, R, t, rmsd


def kabsch_align(X: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords_wt_aligned, rotation, translation, _ = kabsch_transform(X, Y, mask)
    return coords_wt_aligned.astype(np.float32), rotation.astype(np.float32), translation.astype(np.float32)


def write_ca_only_pdb(coords: np.ndarray, path: Path, chain_id: str = "A") -> None:
    lines = []
    for index, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {index:5d}  CA  GLY {chain_id}{index:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_numeric_matrix_lines(text: str) -> tuple[np.ndarray, np.ndarray] | None:
    rows: list[list[float]] = []
    for line in text.splitlines():
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", line)
        if len(numbers) >= 4:
            if len(numbers) >= 5 and numbers[0] in {"1", "2", "3"}:
                values = [float(x) for x in numbers[1:5]]
            else:
                values = [float(x) for x in numbers[:4]]
            rows.append(values)
            if len(rows) == 3:
                t = np.asarray([row[0] for row in rows], dtype=np.float64)
                R = np.asarray([row[1:] for row in rows], dtype=np.float64)
                return R, t
    return None


def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def tmalign_transform(
    coords_wt: np.ndarray,
    coords_mut: np.ndarray,
    tmalign_bin: str,
    sample_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    _validate_inputs(coords_wt, coords_mut)
    tmalign_path = Path(tmalign_bin)
    if not tmalign_path.exists():
        raise FileNotFoundError(f"TM-align binary not found: {tmalign_bin}")

    with tempfile.TemporaryDirectory(prefix="musrnet_tmalign_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        wt_pdb = tmpdir_path / "wt_ca.pdb"
        mut_pdb = tmpdir_path / "mut_ca.pdb"
        matrix_file = tmpdir_path / "matrix.txt"
        write_ca_only_pdb(coords_wt, wt_pdb)
        write_ca_only_pdb(coords_mut, mut_pdb)

        completed = _run_command([str(tmalign_path), str(wt_pdb), str(mut_pdb), "-m", str(matrix_file)])
        parse_source = ""
        if completed.returncode == 0 and matrix_file.exists():
            parse_source = matrix_file.read_text(encoding="utf-8", errors="replace")
        else:
            retry = _run_command([str(tmalign_path), str(wt_pdb), str(mut_pdb)])
            completed = retry
            parse_source = (retry.stdout or "") + "\n" + (retry.stderr or "")

        parsed = _parse_numeric_matrix_lines(parse_source)
        if parsed is None:
            snippet = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()[:1000]
            raise RuntimeError(f"Failed to parse TM-align matrix for sample {sample_id or '<unknown>'}: {snippet}")

        R, t = parsed
        coords_wt_aligned = coords_wt @ R.T + t
        rmsd = float(np.sqrt(np.mean(np.sum((coords_wt_aligned - coords_mut) ** 2, axis=1))))
        metadata = {
            "external_aligner": "tmalign",
            "tmalign_bin": str(tmalign_path),
            "sample_id": sample_id,
            "stdout": (completed.stdout or "")[:1000],
            "stderr": (completed.stderr or "")[:1000],
        }
        return coords_wt_aligned, R, t, rmsd, metadata


def align_by_variant(
    coords_wt: np.ndarray,
    coords_mut: np.ndarray,
    mut_pos: int,
    variant: str,
    tmalign_bin: str | None = None,
    sample_id: str | None = None,
) -> AlignmentResult:
    _validate_inputs(coords_wt, coords_mut)
    if not (0 <= mut_pos < coords_wt.shape[0]):
        raise ValueError("Mutation position is out of range")

    mut_coord = coords_wt[mut_pos]
    radii = np.linalg.norm(coords_wt - mut_coord[None, :], axis=1)

    if variant == "kabsch_exclude_4A":
        mask = radii > 4.0
        coords_wt_aligned, rotation, translation, rmsd = kabsch_transform(coords_wt, coords_mut, mask)
        metadata = {}
    elif variant == "kabsch_exclude_8A":
        mask = radii > 8.0
        coords_wt_aligned, rotation, translation, rmsd = kabsch_transform(coords_wt, coords_mut, mask)
        metadata = {}
    elif variant == "kabsch_all":
        mask = np.ones(coords_wt.shape[0], dtype=bool)
        coords_wt_aligned, rotation, translation, rmsd = kabsch_transform(coords_wt, coords_mut, mask)
        metadata = {}
    elif variant == "tmalign":
        if not tmalign_bin:
            raise FileNotFoundError("TM-align variant requested but no --tmalign-bin was provided")
        mask = np.ones(coords_wt.shape[0], dtype=bool)
        coords_wt_aligned, rotation, translation, rmsd, metadata = tmalign_transform(
            coords_wt=coords_wt,
            coords_mut=coords_mut,
            tmalign_bin=tmalign_bin,
            sample_id=sample_id,
        )
    else:
        raise ValueError(f"Unsupported alignment variant: {variant}")

    if mask.sum() < 3:
        raise ValueError("Alignment mask must contain at least 3 residues")

    return AlignmentResult(
        coords_wt_aligned=coords_wt_aligned.astype(np.float32),
        rotation=rotation.astype(np.float32),
        translation=translation.astype(np.float32),
        mask=mask.astype(bool),
        alignment_name=variant,
        alignment_rmsd=float(rmsd),
        n_aligned_residues=int(mask.sum()),
        metadata=metadata,
    )
