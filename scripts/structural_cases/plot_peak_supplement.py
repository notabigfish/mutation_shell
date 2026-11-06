"""Export, render and assemble the mutation-proximal peak-recovery example."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpecFromSubplotSpec
from PIL import Image
import torch
from common import OUT, MODEL_PATHS, selected_prediction_rows

CASE = OUT / 'cases/case_peak_supplement'
COLORS = {'true': '#18181B', 'musrnet': '#BE123C', 'geometry': '#0D9488'}
CMAP = LinearSegmentedColormap.from_list('displacement', [
    (0.0, '#F8FAFC'), (0.2, '#FEF08A'), (0.4, '#F97316'), (1.0, '#BE123C')])


def export(pdb_dir):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from musrnet.alignment import kabsch_transform
    from musrnet.pdb_io import inspect_chain
    from export_case_data import ordered_residues, save_aligned_chain
    choice = pd.read_csv(OUT / 'selected_peak_supplement.csv', dtype={'sample_id': str}).iloc[0]
    sid = str(choice.sample_id)
    sample = torch.load(Path('data/processed/samples') / f'{sid}.pt', map_location='cpu', weights_only=False)
    n = len(sample['wt_sequence'])
    CASE.mkdir(parents=True, exist_ok=True)
    frames = {name: selected_prediction_rows(MODEL_PATHS[name], {sid}).sort_values('residue_index')
              for name in ('musrnet', 'geometry')}
    data = pd.DataFrame({'residue_index': np.arange(n), 'position': np.arange(1,n+1),
                         'true': sample['displacement'].numpy(),
                         'true_perturbed': sample['perturbed'].numpy(),
                         'radii': sample['radii'].numpy(), 'shell_id': sample['shell_id'].numpy()})
    for name, f in frames.items():
        assert len(f) == n and f.sample_id.nunique() == 1
        data[name] = f.pred_displacement.to_numpy(float)
        data[f'prob_{name}'] = f.pred_perturbed_prob.to_numpy(float)
    data.to_csv(CASE / 'residues.csv', index=False)
    chains = {}
    for tag in ('wt', 'mut'):
        pdb_id = str(sample[f'{tag}_pdb_id']).lower()
        info = inspect_chain(pdb_dir / f'{pdb_id}.cif.gz', sample[f'{tag}_chain_id'],
                             'mmcif', str(pdb_dir.parent.parent), pdb_dir.parent.name)
        assert info['sequence'] == sample[f'{tag}_sequence']
        chains[tag] = ordered_residues(info['chain_df'], info['residues'])
    wt = np.asarray([r['CA'].coord for r in chains['wt']]); mut = np.asarray([r['CA'].coord for r in chains['mut']])
    assert len(wt) == len(mut) == n
    assert np.max(np.abs(wt - sample['coords_wt'].numpy())) < .02
    assert np.max(np.abs(mut - sample['coords_mut'].numpy())) < .02
    aligned, rotation, translation, _ = kabsch_transform(wt, mut, sample['radii'].numpy()>4.)
    assert np.max(np.abs(np.linalg.norm(aligned - mut, axis=1) - data.true)) < .02
    save_aligned_chain(chains['wt'], CASE/'wt.pdb', rotation, translation)
    save_aligned_chain(chains['mut'], CASE/'mut.pdb')
    for field in ('true', 'musrnet', 'geometry'):
        save_aligned_chain(chains['wt'], CASE/f'wt_{field}.pdb', rotation, translation,
                           data[field].to_numpy(float))
    metadata = {'sample_id':sid, 'wt_pdb_id':sample['wt_pdb_id'], 'wt_chain':sample['wt_chain_id'],
                'mut_pdb_id':sample['mut_pdb_id'], 'mut_chain':sample['mut_chain_id'],
                'mutation_resi':int(sample['mut_pos'])+1, 'wt_aa':sample['wt_aa'],
                'mut_aa':sample['mut_aa'], 'true_peak_radius_A':float(choice.true_peak_radius_A),
                'peak_relative_error':float(choice.local_peak_relative_error),
                'auprc':float(choice.auprc), 'geometry_auprc':float(choice.geometry_auprc)}
    (CASE/'metadata.json').write_text(json.dumps(metadata, indent=2))
    write_pml(CASE)


def write_pml(path):
    from write_pymol_scripts import HEADER
    m = json.loads((path/'metadata.json').read_text())
    code = f'''root=Path({str(path.resolve())!r})
rows=read_rows(root/'residues.csv')
site={m['mutation_resi']}
peak=max(range(1,len(rows)+1),key=lambda i:float(rows[i-1]['true']))
cmd.load(str(root/'wt.pdb'),'wt')
cmd.load(str(root/'mut.pdb'),'mut')
cmd.show('cartoon','wt or mut')
cmd.color('slate','wt');cmd.color('gray70','mut')
cmd.set('cartoon_transparency',.45,'wt')
cmd.set_color('observed_shift',ROSE)
for i,row in enumerate(rows,1):
    if float(row['true'])>1.0: cmd.color('observed_shift','mut and resi '+str(i))
cmd.show('sticks',f'wt and resi {{site}}')
cmd.show('sticks',f'mut and resi {{site}}')
cmd.orient('wt or mut')
cmd.zoom('wt or mut',buffer=3.0)
view=cmd.get_view()
p0=cmd.get_atom_coords(f'wt and resi {{peak}} and name CA')
p1=cmd.get_atom_coords(f'mut and resi {{peak}} and name CA')
arrow(p0,p1,'true_peak_arrow')
cmd.set_view(view)
save(str(root/'supp_overlay.png'))
cmd.delete('all')
for field in ('true','musrnet','geometry'):
    cmd.load(str(root/('wt_'+field+'.pdb')),'response')
    cmd.show('cartoon','response')
    paint('response',rows,field)
    cmd.show('spheres',f'response and resi {{site}} and name CA')
    cmd.set('sphere_scale',.36,'response')
    cmd.set_view(view)
    save(str(root/('supp_'+field+'.png')))
    cmd.delete('all')
'''
    pml = OUT/'pymol/peak_supplement.pml'
    pml.parent.mkdir(parents=True, exist_ok=True)
    pml.write_text(HEADER + code + '\npython end\n')
    print('PyMOL input:',pml)


def image(ax, path, title):
    im = Image.open(path).convert('RGBA')
    pixels = np.asarray(im)
    if np.any(pixels[:,:,3] < 254):
        mask = pixels[:,:,3] > 5
    else:
        mask = np.any(pixels[:,:,:3] < 240, axis=2)
    yy, xx = np.nonzero(mask)
    margin = int(.04 * max(xx.max()-xx.min(), yy.max()-yy.min()))
    crop = im.crop((max(0,int(xx.min())-margin),max(0,int(yy.min())-margin),
                    min(im.width,int(xx.max())+margin),min(im.height,int(yy.max())+margin)))
    ax.imshow(crop)
    ax.axis('off')
    ax.set_title(title, fontsize=12, pad=6)


def compose():
    d = pd.read_csv(CASE/'residues.csv')
    m = json.loads((CASE/'metadata.json').read_text())
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                         'font.size':11,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,
                         'pdf.fonttype':42})
    fig = plt.figure(figsize=(12.5,7.0),facecolor='white')
    gs = fig.add_gridspec(3,4,height_ratios=[1.65,.11,1.15],left=.065,right=.975,
                          top=.885,bottom=.10,hspace=.30,wspace=.20)
    for j,(fn,title) in enumerate((('supp_overlay.png','WT / mutant'),('supp_true.png','Observed'),
                                    ('supp_musrnet.png','MuSRNet'),('supp_geometry.png','Geometry GNN'))):
        ax=fig.add_subplot(gs[0,j]);image(ax,CASE/fn,title)
    cb_ax=fig.add_subplot(gs[1,1:])
    cbar=fig.colorbar(ScalarMappable(norm=Normalize(0,2.5),cmap=CMAP),cax=cb_ax,orientation='horizontal')
    cbar.set_ticks([0,.5,1,2,2.5]);cbar.set_label(r'C$\alpha$ displacement ($\mathrm{\AA}$)',fontsize=10)
    bottom=GridSpecFromSubplotSpec(1,2,subplot_spec=gs[2,:],wspace=.34)
    ax=fig.add_subplot(bottom[0]); ap=fig.add_subplot(bottom[1])
    for field,style in (('true','-'),('musrnet','-'),('geometry','--')):
        ax.plot(d.position,d[field],style,color=COLORS[field],lw=1.5,label=field.title())
    ax.axhline(1.0,color='#AAA',ls=':',lw=.8)
    ax.axvline(m['mutation_resi'],color='#D97706',ls=':',lw=1)
    ax.set(xlabel='Residue position',ylabel=r'C$\alpha$ displacement ($\mathrm{\AA}$)')
    ax.legend(frameon=False,ncol=3,fontsize=9,loc='upper right')
    for field,style in (('musrnet','-'),('geometry','--')):
        ap.plot(d.position,d[f'prob_{field}'],style,color=COLORS[field],lw=1.3,label=field.title())
    ap.fill_between(d.position,0,d.true_perturbed,alpha=.13,color='#D97706',label='Observed perturbed')
    ap.axvline(m['mutation_resi'],color='#D97706',ls=':',lw=1)
    ap.set(xlabel='Residue position',ylabel='Perturbed-residue probability')
    ap.set_ylim(0, min(1.0, max(0.2, 1.1 * max(d.prob_musrnet.max(), d.prob_geometry.max()))))
    ap.legend(frameon=False,fontsize=8,ncol=2,loc='upper right')
    for axis in (ax,ap):axis.spines[['top','right']].set_visible(False)
    fig.suptitle(f"Mutation-proximal peak case: {m['sample_id']}   |   "
                 f"AUPRC {m['auprc']:.2f} vs. {m['geometry_auprc']:.2f}",fontsize=14)
    fig.savefig(OUT/'figure_peak_recovery_supplement.pdf',dpi=300)
    fig.savefig(OUT/'figure_peak_recovery_supplement.png',dpi=300)
    plt.close(fig)
    print('Wrote figure_peak_recovery_supplement.pdf')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('export','compose'))
    p.add_argument('--pdb-dir',type=Path,default=Path('data/pdb'))
    a=p.parse_args()
    export(a.pdb_dir) if a.command=='export' else compose()


if __name__=='__main__':
    main()
