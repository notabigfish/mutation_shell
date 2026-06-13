# MuSRNet

MuSRNet predicts residue-level mutation-induced structural response from paired wild-type and mutant protein structures. The pipeline covers preprocessing, structural label construction, frozen ESM-2 embedding precomputation, graph-based training, and evaluation.

## Input data

The main CSV file is `data/SingleMutPairs2024.csv` with columns:

```python
[
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
```

PDB/mmCIF files must be available under `data/pdb/` as lowercase `{pdb_id}.cif.gz`.

## Target definition

For a sample with wild-type sequence `S^{WT}` and mutant sequence `S^{Mut}`, wild-type C-alpha coordinates `X=(x_1,\ldots,x_L)` and mutant coordinates `Y=(y_1,\ldots,y_L)`, the mutation is `m=(p,a_{WT},a_{Mut})` where `p=mut_pos_seq_index`.

Each sample is validated by:

```math
S^{WT}_p=a_{WT}, \qquad S^{Mut}_p=a_{Mut}, \qquad \operatorname{Hamming}(S^{WT}, S^{Mut})=1.
```

Using residues farther than 4 Angstrom from the mutation site, MuSRNet computes a Kabsch alignment:

```math
A=\{i:\|x_i-x_p\|_2>4.0\},
```

```math
(R^*,t^*)=\arg\min_{R\in SO(3),t\in\mathbb{R}^3}\sum_{i\in A}\|Rx_i+t-y_i\|_2^2.
```

Aligned wild-type coordinates are `\tilde{x}_i = R^*x_i + t^*`, and the residue displacement label is:

```math
d_i=\|\tilde{x}_i-y_i\|_2.
```

Residue distance to the mutation site is:

```math
r_i=\|x_i-x_p\|_2.
```

Shells are:

```math
\mathcal{S}_0=\{i:r_i\le4\},
\mathcal{S}_1=\{i:4<r_i\le8\},
\mathcal{S}_2=\{i:8<r_i\le12\},
\mathcal{S}_3=\{i:12<r_i\le16\},
\mathcal{S}_4=\{i:r_i>16\}.
```

With threshold `\tau=1.0` Angstrom, the perturbed label is:

```math
z_i=\mathbf{1}[d_i>\tau].
```

The perturbation radius is:

```math
\rho=\max_{i:z_i=1} r_i,
```

with `\rho=0` if no residue is perturbed. The response class is:

```math
c=
\begin{cases}
0 & \max_i d_i \le 1.0 \quad \text{silent}\\
1 & \rho \le 8.0 \quad \text{local}\\
2 & \rho > 8.0 \quad \text{non-local}
\end{cases}
```

## Shell-balanced loss theorem

Theorem: Shell-balanced loss prevents mutation-signal dilution.

Let the residue loss be `ell_i`. Standard global loss is:

```math
\mathcal{L}_{global}=\frac{1}{L}\sum_{i=1}^{L}\ell_i.
```

If the true perturbed residue set is `P` and `|P|=q << L`, then the total loss weight assigned to perturbed residues is:

```math
\frac{q}{L}.
```

Thus, as `q/L -> 0`, the mutation-relevant signal vanishes.

MuSRNet uses shell-balanced loss:

```math
\mathcal{L}_{shell}
=
\frac{1}{5}
\sum_{k=0}^{4}
\frac{1}{|\mathcal{S}_k|}
\sum_{i\in\mathcal{S}_k}
H_\delta(\hat d_i-d_i).
```

The total coefficient of each shell is:

```math
\sum_{i\in\mathcal{S}_k}
\frac{1}{5|\mathcal{S}_k|}
=
\frac{1}{5}.
```

Therefore, small mutation-proximal shells cannot be overwhelmed by large distal shells. This is the core mechanism of MuSRNet.

## Env

```bash
conda activate pt311cu130
cd /rds/projects/l/liuje-multiai/shuo/mutation/MuSRNet
```

## Preprocessing

```bash
python process_data.py --output_dir data/ --pdb_version pdb_260603 --pdb_format mmcif --re_subset --n_clusters 1000
```

```bash
python scripts/prepare_data.py --csv data/SingleMutPairs2024_subset_c1000.csv  --out data/processed/samples_subset_c1000_raw.pt
```

This creates `data/processed/samples.pt` as a manifest and stores one processed sample dictionary per file under `data/processed/samples/`.

## ESM precomputation

```bash
python scripts/precompute_esm.py --samples data/processed/samples_subset_c1000_raw.pt --out-esm-lmdb data/processed/esm_subset_c1000.lmdb --out-filtered-manifest data/processed/samples_subset_c1000.pt
```

## Training

```bash
python scripts/train.py --config configs/c1000/base_v1.yaml
python scripts/train.py --config configs/c1000/base_v2.yaml
```

## Evaluation

```bash
python scripts/evaluate.py \
    --config configs/musrnet.yaml \
    --checkpoint outputs/best.pt
```

## Outputs

- `data/processed/samples.pt`
- `data/processed/samples/*.pt`
- `data/processed/esm/*.pt`
- `data/processed/splits.json`
- `outputs/best.pt`
- `outputs/last.pt`
- `outputs/train_log.csv`
- `outputs/config_used.yaml`
- `outputs/eval_train.json`
- `outputs/eval_valid.json`
- `outputs/eval_test.json`
- `outputs/predictions_test.csv`
