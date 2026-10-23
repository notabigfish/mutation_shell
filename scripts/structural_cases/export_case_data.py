"""Export four aligned full-atom structure pairs and model fields for rendering.

Uses existing prediction CSVs; only the MuSRNet and Coordinate checkpoints
are re-evaluated for outputs absent from the saved CSVs.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB import MMCIFParser, PDBIO
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from musrnet.alignment import kabsch_transform
from musrnet.dataset import MuSRNetDataset, load_samples_manifest
from musrnet.models import build_model
from musrnet.pdb_io import inspect_chain
from musrnet.train_utils import load_yaml
from common import OUT, MODEL_PATHS, selected_prediction_rows


def read_cif(path: Path, chain_id: str):
    with gzip.open(path, 'rt') as fh:
        structure = MMCIFParser(QUIET=True).get_structure('original', fh)
    return structure[0][chain_id]


def ordered_residues(chain, residue_info):
    atoms = [r for r in chain.get_residues() if 'CA' in r]
    chosen = []
    for info in residue_info:
        hit = next(r for r in atoms if int(r.id[1]) == int(info['pdb_number'])
                   and str(r.id[2]).strip() == str(info['icode']))
        chosen.append(hit)
    return chosen


def save_aligned_chain(residues, path, rotation=None, translation=None, b=None):
    structure = Structure('case')
    model = Model(0)
    chain = Chain('A')
    structure.add(model)
    model.add(chain)
    for i, r in enumerate(residues):
        new = copy.deepcopy(r)
        new.id = (' ', i + 1, ' ')
        for atom in new:
            if rotation is not None:
                atom.set_coord(atom.coord @ rotation.T + translation)
            if b is not None:
                atom.set_bfactor(float(b[i]))
        chain.add(new)
    out = PDBIO()
    out.set_structure(structure)
    out.save(str(path))


def infer_extra(sample_id: str, config_path: str, checkpoint_path: Path):
    config = load_yaml(config_path)
    model = build_model(config['model_name'], config['model'])
    model.load_state_dict(load_file(str(checkpoint_path), device='cpu'))
    model.eval()
    manifest = load_samples_manifest(config['paths']['samples'])
    dataset = MuSRNetDataset(manifest, [sample_id], config['data']['knn_k'],
                            edge_feature_version=config['data']['edge_feature_version'])
    data = dataset[0]
    inputs = dict(x_basic=data.x_basic, esm_wt=data.esm_wt, esm_delta=data.esm_delta,
                  edge_index=data.edge_index, edge_attr=data.edge_attr, shell_id=data.shell_id,
                  radii=data.radii, is_mutation_site=data.is_mutation_site, mut_pos=data.mut_pos)
    with torch.no_grad():
        result = model(**inputs)
    return {name: tensor.detach().numpy() for name, tensor in result.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb-dir', type=Path, default=Path('data/pdb'))
    args = parser.parse_args()
    selected = pd.read_csv(OUT / 'selected_cases.csv', dtype={'sample_id': str})
    ids = set(selected.sample_id)
    predictions = {name: selected_prediction_rows(MODEL_PATHS[name], ids).set_index(['sample_id', 'residue_index'])
                   for name in MODEL_PATHS}
    fold = pd.read_csv(OUT / 'fold_residue_fields.csv', dtype={'sample_id': str})
    fold = fold[fold.sample_id.isin(ids)].set_index(['sample_id', 'model', 'residue_index'])
    manifest = load_samples_manifest('data/processed/samples.pt')
    annotations = []

    for row in selected.itertuples(index=False):
        case_dir = OUT / 'cases' / row.case
        case_dir.mkdir(parents=True, exist_ok=True)
        s = torch.load(Path(manifest['samples_dir']) / f'{row.sample_id}.pt', map_location='cpu', weights_only=False)
        L = len(s['wt_sequence'])
        frame = pd.DataFrame({
            'residue_index': np.arange(L), 'position': np.arange(1, L + 1),
            'true': s['displacement'].numpy(), 'true_perturbed': s['perturbed'].numpy(),
            'radii': s['radii'].numpy(), 'shell_id': s['shell_id'].numpy(),
        })
        for name, pred in predictions.items():
            this = pred.loc[row.sample_id].sort_index()
            assert len(this) == L, (row.sample_id, name)
            frame[name] = this.pred_displacement.to_numpy(float)
            frame[f'prob_{name}'] = this.pred_perturbed_prob.to_numpy(float)
        for model in ('af2', 'esmfold'):
            this = fold.loc[(row.sample_id, model)].sort_index()
            frame[model] = this.pred_displacement.to_numpy(float)
        frame['score'] = frame.prob_musrnet * frame.musrnet
        frame['abs_error_full'] = np.abs(frame.musrnet - frame.true)
        frame['abs_error_global'] = np.abs(frame.global_loss - frame.true)
        frame['abs_error_direct'] = np.abs(frame.direct - frame.true)

        if row.case == 'case1_comparison':
            extra = infer_extra(row.sample_id, 'configs/call/coordinate_residual.yaml',
                                Path('outputs/call/coordinate_residual/best/model.safetensors'))
            vector = extra['coord_residual']
            frame[['vector_x', 'vector_y', 'vector_z']] = vector
            assert np.max(np.abs(np.linalg.norm(vector, axis=1) - frame.coordinate)) < 0.005
        if row.case == 'case4_mechanism':
            extra = infer_extra(row.sample_id, 'configs/call/base_v5_seed42.yaml',
                                Path('outputs/call/base_v5_seed42/best/model.safetensors'))
            frame['background'] = extra['background']
            frame['gated_excess'] = (1 / (1 + np.exp(-extra['perturbed_logit']))) * extra['excess']
            frame['log_variance'] = extra['disp_logvar']
            assert np.max(np.abs(extra['disp'] - frame.musrnet)) < 0.005
        frame.to_csv(case_dir / 'residues.csv', index=False)

        wt_file = args.pdb_dir / f'{s["wt_pdb_id"]}.cif.gz'
        mut_file = args.pdb_dir / f'{s["mut_pdb_id"]}.cif.gz'
        pdb_root = str(args.pdb_dir.parent.parent)  # layout: <root>/<version>/pdb
        pdb_version = args.pdb_dir.parent.name
        wt_info = inspect_chain(wt_file, s['wt_chain_id'], 'mmcif', pdb_root, pdb_version)
        mut_info = inspect_chain(mut_file, s['mut_chain_id'], 'mmcif', pdb_root, pdb_version)
        wt_res = ordered_residues(read_cif(wt_file, s['wt_chain_id']), wt_info['residues'])
        mut_res = ordered_residues(read_cif(mut_file, s['mut_chain_id']), mut_info['residues'])
        assert len(wt_res) == len(mut_res) == L
        wt_ca = np.array([r['CA'].coord for r in wt_res]); mut_ca = np.array([r['CA'].coord for r in mut_res])
        assert np.max(np.abs(wt_ca - s['coords_wt'].numpy())) < 0.01
        assert np.max(np.abs(mut_ca - s['coords_mut'].numpy())) < 0.01
        aligned, rotation, translation, _ = kabsch_transform(wt_ca, mut_ca, s['radii'].numpy() > 4.0)
        assert np.max(np.abs(aligned - s['coords_wt_aligned'].numpy())) < 0.01
        save_aligned_chain(wt_res, case_dir / 'wt.pdb', rotation, translation)
        save_aligned_chain(mut_res, case_dir / 'mut.pdb')
        for field in ('true', 'musrnet', 'geometry', 'coordinate', 'af2', 'esmfold',
                      'background', 'gated_excess', 'abs_error_full', 'abs_error_global'):
            if field in frame:
                save_aligned_chain(wt_res, case_dir / f'wt_{field}.pdb', rotation, translation,
                                   frame[field].to_numpy(float))
        metadata = {'case': row.case, 'sample_id': row.sample_id,
                    'wt_pdb_id': s['wt_pdb_id'], 'wt_chain': s['wt_chain_id'],
                    'mut_pdb_id': s['mut_pdb_id'], 'mut_chain': s['mut_chain_id'],
                    'wt_aa': s['wt_aa'], 'mut_aa': s['mut_aa'],
                    'mutation_index': int(s['mut_pos']), 'mutation_resi': int(s['mut_pos']) + 1,
                    'wt_pdb_number': s['wt_pos_pdb_number'], 'mut_pdb_number': s['mut_pos_pdb_number'],
                    'length': L}
        (case_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        annotations.append(metadata)
        print(f'Exported {row.case}: {row.sample_id}')
    pd.DataFrame(annotations).to_csv(OUT / 'case_structure_metadata.csv', index=False)


if __name__ == '__main__':
    main()
