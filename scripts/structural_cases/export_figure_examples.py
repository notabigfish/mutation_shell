"""Export and render the reselected non-local case and peak-recovery supplement.

Commands:
  python .../export_figure_examples.py prepare --pdb-dir <deposited mmCIF directory>
  pymol -cq results/structural_cases/pymol/case3_reselected.pml
  pymol -cq results/structural_cases/pymol/peak_supplement.pml
  python .../export_figure_examples.py compose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from musrnet.alignment import kabsch_transform
from musrnet.dataset import load_samples_manifest
from musrnet.pdb_io import inspect_chain
from common import OUT, MODEL_PATHS, selected_prediction_rows
from export_case_data import ordered_residues, save_aligned_chain
from write_pymol_scripts import HEADER, radial_code


def export_pair(key: str, row: pd.Series, pdb_dir: Path,
                predictions: dict[str, pd.DataFrame], manifest: dict) -> None:
    sid = str(row.sample_id)
    s = torch.load(Path(manifest['samples_dir']) / f'{sid}.pt', map_location='cpu', weights_only=False)
    L = len(s['wt_sequence'])
    case_dir = OUT / 'cases' / key
    case_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        'residue_index': np.arange(L), 'position': np.arange(1, L+1),
        'true': s['displacement'].numpy(), 'true_perturbed': s['perturbed'].numpy(),
        'radii': s['radii'].numpy(), 'shell_id': s['shell_id'].numpy(),
    })
    for model, pred in predictions.items():
        g = pred.loc[sid].sort_index()
        assert len(g) == L
        frame[model] = g.pred_displacement.to_numpy(float)
        frame[f'prob_{model}'] = g.pred_perturbed_prob.to_numpy(float)
    frame['score'] = frame.prob_musrnet * frame.musrnet
    frame.to_csv(case_dir / 'residues.csv', index=False)

    chain_info = {}
    for tag in ('wt','mut'):
        pdb_id = str(s[f'{tag}_pdb_id']).lower()
        path = pdb_dir / f'{pdb_id}.cif.gz'
        chain_info[tag] = inspect_chain(path, s[f'{tag}_chain_id'], 'mmcif',
                                        str(pdb_dir.parent.parent), pdb_dir.parent.name)
    wt_res = ordered_residues(chain_info['wt']['chain_df'], chain_info['wt']['residues'])
    mut_res = ordered_residues(chain_info['mut']['chain_df'], chain_info['mut']['residues'])
    assert len(wt_res) == len(mut_res) == L
    wt = np.asarray([r['CA'].coord for r in wt_res])
    mut = np.asarray([r['CA'].coord for r in mut_res])
    assert np.max(np.abs(wt - s['coords_wt'].numpy())) < .02
    assert np.max(np.abs(mut - s['coords_mut'].numpy())) < .02
    aligned, R, t, _ = kabsch_transform(wt, mut, s['radii'].numpy() > 4.)
    assert np.max(np.abs(np.linalg.norm(aligned - mut, axis=1) - frame.true.to_numpy())) < .02
    save_aligned_chain(wt_res, case_dir / 'wt.pdb', R, t)
    save_aligned_chain(mut_res, case_dir / 'mut.pdb')
    for field in ('true', 'musrnet', 'geometry'):
        save_aligned_chain(wt_res, case_dir / f'wt_{field}.pdb', R, t, frame[field].to_numpy())
    meta = {
        'case': key, 'sample_id': sid, 'length': L,
        'wt_pdb_id': s['wt_pdb_id'], 'wt_chain_id': s['wt_chain_id'],
        'mut_pdb_id': s['mut_pdb_id'], 'mut_chain_id': s['mut_chain_id'],
        'wt_aa': s['wt_aa'], 'mut_aa': s['mut_aa'], 'mutation_index': int(s['mut_pos']),
        'wt_pdb_number': s['wt_pos_pdb_number'], 'mut_pdb_number': s['mut_pos_pdb_number'],
        'auprc': float(row.auprc), 'geometry_auprc': float(row.geometry_auprc),
        'shell_mae': float(row.shell_mae), 'max_true_displacement': float(row.max_true_displacement),
    }
    (case_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print('Exported:', key, sid)


def peak_pymol_code(folder: Path) -> str:
    return f'''root=Path({str(folder.resolve())!r})
rows=read_rows(root/'residues.csv')
site=next(i for i,r in enumerate(rows,1) if float(r['radii'])<1.e-4)
peak=max(range(1,len(rows)+1),key=lambda i:float(rows[i-1]['true']))
cmd.load(str(root/'wt.pdb'),'wt')
cmd.load(str(root/'mut.pdb'),'mut')
cmd.show('cartoon','wt or mut')
cmd.color('gray70','mut')
cmd.color('slate','wt')
cmd.set('cartoon_transparency',.5,'wt')
for i,r in enumerate(rows,1):
    if float(r['true'])>1.0: cmd.color('firebrick','mut and resi '+str(i))
cmd.show('sticks','wt and resi '+str(site))
cmd.show('sticks','mut and resi '+str(site))
cmd.orient('wt or mut')
view=cmd.get_view()
start=cmd.get_atom_coords('wt and resi '+str(peak)+' and name CA')
end=cmd.get_atom_coords('mut and resi '+str(peak)+' and name CA')
arrow(start,end,'true_peak_arrow')
cmd.set_view(view)
save(str(root/'supp_overlay.png'))
cmd.delete('all')
for field in ('true','musrnet','geometry'):
    cmd.load(str(root/('wt_'+field+'.pdb')),'response')
    cmd.show('cartoon','response')
    paint('response',rows,field)
    cmd.show('spheres','response and resi '+str(site)+' and name CA')
    cmd.set('sphere_scale',.37,'response')
    cmd.set_view(view)
    save(str(root/('supp_'+field+'.png')))
    cmd.delete('all')
'''


def update_case3_selection(new_row: pd.Series) -> None:
    path = OUT / 'selected_cases.csv'
    selected = pd.read_csv(path, dtype={'sample_id': str, 'cluster_id_30': str})
    target = selected.index[selected.case == 'case3_nonlocal']
    assert len(target) == 1
    # Reconstruct this row so no metadata from the discarded example survives.
    new = {c: np.nan for c in selected.columns}
    for col in selected.columns:
        if col in new_row.index:
            new[col] = new_row[col]
    new['case'] = 'case3_nonlocal'
    s = torch.load(Path('data/processed/samples') / f'{new_row.sample_id}.pt',
                   map_location='cpu', weights_only=False)
    for col in ('wt_pdb_id','wt_chain_id','mut_pdb_id','mut_chain_id', 'wt_aa', 'mut_aa'):
        if col in new:
            new[col] = s[col]
    annot_path = Path('outputs/call/biological_stratification/v5_seed42_vs_geometry_gnn/biological_annotations.csv')
    ann = pd.read_csv(annot_path, dtype={'sample_id': str})
    ann = ann[ann.sample_id == str(new_row.sample_id)].iloc[0]
    for col in selected.columns:
        if col in ann.index:
            new[col] = ann[col]
    for suffix, model in (('geometry', 'geometry'), ('coordinate', 'coordinate'),
                          ('global', 'global_loss'), ('direct', 'direct')):
        metrics = pd.read_csv(OUT / f'case_sample_metrics_{model}.csv', dtype={'sample_id': str})
        record = metrics.loc[metrics.sample_id == str(new_row.sample_id)].iloc[0]
        for name in ('auprc', 'shell_mae', 'near_mae'):
            col = f'{name}_{suffix}'
            if col in new:
                new[col] = record[name]
    selected.loc[target[0], :] = [new[col] for col in selected.columns]
    selected.to_csv(path, index=False)


def prepare(pdb_dir: Path):
    choice3 = pd.read_csv(OUT/'selected_nonlocal_replacement.csv', dtype={'sample_id': str}).iloc[0]
    choice_peak = pd.read_csv(OUT/'selected_peak_supplement.csv', dtype={'sample_id': str}).iloc[0]
    sample_ids = {str(choice3.sample_id), str(choice_peak.sample_id)}
    predictions = {name: selected_prediction_rows(MODEL_PATHS[name], sample_ids).set_index(['sample_id','residue_index'])
                   for name in ('musrnet','geometry')}
    manifest = load_samples_manifest('data/processed/samples.pt')
    export_pair('case3_nonlocal', choice3, pdb_dir, predictions, manifest)
    export_pair('case_peak_supplement', choice_peak, pdb_dir, predictions, manifest)
    update_case3_selection(choice3)
    script_dir = OUT/'pymol'
    script_dir.mkdir(parents=True, exist_ok=True)
    (script_dir/'case3_reselected.pml').write_text(
        HEADER + radial_code(OUT/'cases/case3_nonlocal') + '\npython end\n')
    (script_dir/'peak_supplement.pml').write_text(
        HEADER + peak_pymol_code(OUT/'cases/case_peak_supplement') + '\npython end\n')
    print('PyMOL scripts:', script_dir/'case3_reselected.pml', script_dir/'peak_supplement.pml')


def compose():
    path = OUT / 'cases/case_peak_supplement'
    data = pd.read_csv(path/'residues.csv')
    meta = json.loads((path/'metadata.json').read_text())
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial','Helvetica','DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(10,5.7))
    gs = fig.add_gridspec(2,4,height_ratios=[1.2,1],left=.05,right=.98,top=.92,bottom=.10,hspace=.32,wspace=.12)
    labels = [('supp_overlay.png','Experimental WT / mutant'),('supp_true.png','Observed'),
              ('supp_musrnet.png','MuSRNet'),('supp_geometry.png','Geometry GNN')]
    for j,(filename,title) in enumerate(labels):
        ax = fig.add_subplot(gs[0,j]);ax.imshow(plt.imread(path/filename));ax.axis('off');ax.set_title(title,fontsize=9)
    ax = fig.add_subplot(gs[1,:3]);
    ax.plot(data.position,data.true,color='#222222',lw=1.4,label='Experimental')
    ax.plot(data.position,data.musrnet,color='#BE123C',lw=1.3,label='MuSRNet')
    ax.plot(data.position,data.geometry,color='#0D9488',lw=1.0,ls='--',label='Geometry GNN')
    ax.axhline(1.0,color='#AAAAAA',ls=':',lw=1)
    ax.axvline(meta['mutation_index']+1,color='#D97706',ls=':',lw=1)
    ax.set_xlabel('Residue position');ax.set_ylabel(r'C$\alpha$ displacement ($\mathrm{\AA}$)')
    ax.legend(frameon=False,loc='upper right',fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    ap = fig.add_subplot(gs[1,3]);
    ap.plot(data.position,data.prob_musrnet,color='#BE123C',label='MuSRNet')
    ap.plot(data.position,data.prob_geometry,color='#0D9488',ls='--',label='Geometry')
    ap.fill_between(data.position,0,data.true_perturbed,color='#D97706',alpha=.20)
    ap.set_ylabel('Perturbation probability');ap.set_xlabel('Residue position')
    ap.spines[['top','right']].set_visible(False)
    fig.suptitle(f"Peak-recovery case  {meta['sample_id']}  |  MuSRNet AP {meta['auprc']:.2f}; "
                 f"Geometry AP {meta['geometry_auprc']:.2f}",fontsize=11)
    fig.savefig(OUT/'figure_peak_recovery_supplement.pdf',dpi=300)
    fig.savefig(OUT/'figure_peak_recovery_supplement.png',dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare','compose'))
    parser.add_argument('--pdb-dir', type=Path, default=Path('data/pdb'))
    args = parser.parse_args()
    prepare(args.pdb_dir) if args.command == 'prepare' else compose()


if __name__ == '__main__':
    main()
