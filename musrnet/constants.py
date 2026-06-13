from __future__ import annotations

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
STANDARD_AA_3_TO_1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
NUM_SHELLS = 5
SHELL_BOUNDS = (4.0, 8.0, 12.0, 16.0)
PERTURBATION_THRESHOLD = 1.0
ALIGNMENT_EXCLUSION_RADIUS = 4.0
NUM_RBF = 16
RBF_CENTERS = np.linspace(0.0, 32.0, NUM_RBF, dtype=np.float32)
RBF_SIGMA = 2.0
