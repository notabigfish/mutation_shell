"""Assemble publication figures from experimental structures and saved predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from common import OUT

C = {'true':'#18181B','musrnet':'#BE123C','geometry':'#0D9488',
     'coordinate':'#7C3AED','af3':'#94A3B8','esmfold2':'#D97706',
     'global_loss':'#0284C7','direct':'#64748B','pure_hurdle':'#D97706'}
CMAP=LinearSegmentedColormap.from_list('displacement',[
    (0.,'#F8FAFC'),(.2,'#FEF08A'),(.4,'#F97316'),(1.,'#BE123C')])
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                     'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                     'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8,
                     'pdf.fonttype':42,'ps.fonttype':42})


def tidy(ax):
    ax.spines[['top','right']].set_visible(False)


def trim_image(path):
    """Crop transparent/white margins only, preserving the molecular view."""
    im=Image.open(path).convert('RGBA')
    a=np.asarray(im)
    if np.any(a[:,:,3]<254):
        mask=a[:,:,3]>5
    else:
        mask=np.any(a[:,:,:3]<240,axis=2)
    y,x=np.nonzero(mask)
    margin=int(.045*max(x.max()-x.min(),y.max()-y.min()))
    return im.crop((max(0,int(x.min())-margin),max(0,int(y.min())-margin),
                    min(im.width,int(x.max())+margin),min(im.height,int(y.max())+margin)))


def show(ax, path, title=''):
    ax.imshow(trim_image(path))
    ax.axis('off')
    if title: ax.set_title(title,fontsize=10,pad=3)


def letter(ax, value):
    ax.text(-.05,1.09,value,transform=ax.transAxes,fontsize=17,fontweight='bold',
            va='top',ha='right')


def radial(ax, df):
    shell=np.arange(5)
    gb=df.groupby('shell_id')
    for field in ('true','musrnet'):
        means=gb[field].mean().reindex(shell)
        ax.plot(shell,means,marker='o',color=C[field],lw=1.4,label='Observed mean' if field=='true' else 'MuSRNet mean')
    ax.scatter(shell,gb.true.max().reindex(shell),color='#D97706',marker='^',s=28,
               label='Observed maximum',zorder=5)
    ax.set_xticks(shell,['S0','S1','S2','S3','S4'])
    ax.set_xlabel('Radial shell')
    ax.set_ylabel(r'C$\alpha$ displacement ($\mathrm{\AA}$)')
    tidy(ax)
    ax.legend(frameon=False,fontsize=7,loc='upper right')


def main_figure():
    case1=OUT/'cases/case1_comparison'
    x=pd.read_csv(case1/'residues.csv')
    folds=pd.read_csv(OUT/'fold_baseline_per_sample.csv',dtype={'sample_id':str})
    sel=pd.read_csv(OUT/'selected_cases.csv',dtype={'sample_id':str})
    sample_id=sel.loc[sel.case=='case1_comparison','sample_id'].iloc[0]
    case_metrics=folds[folds.sample_id==sample_id].set_index('model')
    metrics=pd.read_csv(OUT/'fold_baseline_summary.csv')
    far=pd.read_csv(OUT/'nonlocal_far_localization.csv').set_index('region').loc['S2-S4']

    fig=plt.figure(figsize=(13.4,9.5),facecolor='white')
    outer=GridSpec(3,1,figure=fig,height_ratios=[1.17,1.02,1.05],hspace=.37,
                   left=.052,right=.98,top=.96,bottom=.09)
    top=GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0],width_ratios=[.37,.63],wspace=.15)
    a=GridSpecFromSubplotSpec(1,2,subplot_spec=top[0],wspace=.07)
    aa=fig.add_subplot(a[0]);ab=fig.add_subplot(a[1])
    show(aa,case1/'overlay.png','WT / mutant')
    show(ab,case1/'zoom.png','Local shift')
    letter(aa,'a')

    b=GridSpecFromSubplotSpec(3,5,subplot_spec=top[1],height_ratios=[1.0,.17,.11],
                               wspace=.025,hspace=.06)
    labels=[('true','Observed'),('musrnet','MuSRNet'),('geometry','Geometry GNN'),
            ('coordinate','Coordinate\nresidual'),('af3','AlphaFold3')]
    for i,(field,name) in enumerate(labels):
        ax=fig.add_subplot(b[0,i]);show(ax,case1/(field+'.png'),name)
        if i==0:letter(ax,'b')
        textax=fig.add_subplot(b[1,i]);textax.axis('off')
        if field != 'true':
            r=case_metrics.loc[field]
            textax.text(.5,.5,f"MAE {r.shell_mae:.2f} | AP {r.auprc:.2f}",
                        ha='center',va='center',fontsize=7.8,transform=textax.transAxes)
    cax=fig.add_subplot(b[2,:])
    cb=fig.colorbar(ScalarMappable(cmap=CMAP,norm=Normalize(0,2.5)),cax=cax,orientation='horizontal')
    cb.set_ticks([0,.5,1,2,2.5]);cb.ax.tick_params(labelsize=8,length=2)
    cb.set_label(r'C$\alpha$ displacement (Å)',fontsize=9,labelpad=1)

    mid=GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=.21)
    c=GridSpecFromSubplotSpec(2,1,subplot_spec=mid[0],height_ratios=[1.4,.6],hspace=.08)
    ac=fig.add_subplot(c[0]);ap=fig.add_subplot(c[1],sharex=ac)
    shades={0:'#FDE68A',1:'#CCFBF1',2:'#DCEAFE'}
    for i,row in x.iterrows():
        k=int(row.shell_id)
        if k in shades:
            for ax in (ac,ap): ax.axvspan(i+.5,i+1.5,color=shades[k],alpha=.17,lw=0)
    for field,style in (('true','-'),('musrnet','-'),('geometry','--'),('coordinate','-.'),('af3',':')):
        ac.plot(x.position,x[field],style,color=C[field],lw=1.1,label=field.replace('_',' ').title())
    ac.axhline(1.,ls=':',color='#888',lw=.9)
    ac.set_ylabel(r'$d_i$ (Å)')
    ac.legend(frameon=False,ncol=3,fontsize=7.5,loc='upper right')
    for field in ('musrnet','geometry','coordinate'):
        ap.plot(x.position,x[f'prob_{field}'],color=C[field],lw=1)
    ap.fill_between(x.position,0,x.true_perturbed,alpha=.11,color='#D97706')
    ap.set_ylabel(r'$\hat p_i$');ap.set_xlabel('Residue position')
    for ax in (ac,ap):tidy(ax)
    letter(ac,'c')

    d=GridSpecFromSubplotSpec(1,3,subplot_spec=mid[1],wspace=.42,width_ratios=[1,1,1.12])
    order=['musrnet','geometry','coordinate','esmfold2','af3']
    short=['MuSRNet','Geometry','Coord.','ESMFold2','AF3']
    for col,metric in enumerate(('shell_mae','auprc')):
        ax=fig.add_subplot(d[col])
        sub=metrics[metrics.metric==metric].set_index('model').loc[order]
        values=sub.cluster_mean.to_numpy(float)
        ax.bar(range(5),values,color=[C[q] for q in order],width=.71)
        ax.errorbar(range(5),values,
                    yerr=[values-sub.ci95_low.to_numpy(float),sub.ci95_high.to_numpy(float)-values],
                    fmt='none',ecolor='#333',capsize=2,lw=.85)
        ax.set_xticks(range(5),short,rotation=50,ha='right',fontsize=8)
        ax.set_title('Cluster shell-MAE' if col==0 else 'Cluster AUPRC',fontsize=10)
        tidy(ax)
        if col==0:letter(ax,'d')
    ax=fig.add_subplot(d[2]);subset=folds[folds.model.isin(('musrnet','af3','esmfold2'))]
    for name in ('musrnet','esmfold2','af3'):
        df=subset[subset.model==name]
        ax.scatter(df.max_true_disp,df.max_pred_disp,color=C[name],s=8,alpha=.45,label=name)
    lim=float(max(subset.max_true_disp.max(),subset.max_pred_disp.quantile(.98)))
    ax.plot([0,lim],[0,lim],ls=':',color='#777',lw=.9)
    ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='Observed max (Å)',ylabel='Predicted max (Å)')
    ax.set_title('Response amplitude',fontsize=10)
    ax.legend(frameon=False,fontsize=7,loc='upper right');tidy(ax)

    bottom=GridSpecFromSubplotSpec(1,2,subplot_spec=outer[2],wspace=.22)
    for col,case,panel in ((0,'case2_local','e'),(1,'case3_nonlocal','f')):
        inner=GridSpecFromSubplotSpec(1,2,subplot_spec=bottom[col],width_ratios=[1.10,1.0],wspace=.12)
        img=fig.add_subplot(inner[0]);chart=fig.add_subplot(inner[1])
        folder=OUT/'cases'/case
        show(img,folder/'radial.png','Local response' if col==0 else 'Non-local response')
        radial(chart,pd.read_csv(folder/'residues.csv'))
        if col==1:
            chart.text(.5,-.31,
                       f"S2–S4: Precision {far.precision:.2f}   Recall {far.recall:.2f}   Jaccard {far.jaccard:.2f}",
                       transform=chart.transAxes,ha='center',va='top',fontsize=8.3)
        letter(img,panel)
    fig.savefig(OUT/'figure_structural_cases.pdf',dpi=300)
    fig.savefig(OUT/'figure_structural_cases.png',dpi=300)
    plt.close(fig)
    print('Wrote figure_structural_cases.pdf')


def mechanism_figure():
    case=OUT/'cases/case4_mechanism'
    x=pd.read_csv(case/'residues.csv')
    fig=plt.figure(figsize=(11,4.5),facecolor='white')
    gs=GridSpec(1,3,figure=fig,left=.035,right=.975,bottom=.19,top=.90,wspace=.20,width_ratios=[1.34,1.0,1.16])
    a=GridSpecFromSubplotSpec(1,4,subplot_spec=gs[0,0],wspace=.015)
    for i,(field,title) in enumerate([('background','Background'),('gated_excess','Gated excess'),
                                      ('musrnet','Combined'),('true','Observed')]):
        ax=fig.add_subplot(a[i]);show(ax,case/(field+'.png'),title)
        if i==0:letter(ax,'a')
    b=GridSpecFromSubplotSpec(1,2,subplot_spec=gs[0,1],wspace=.02)
    for i,(field,title) in enumerate([('abs_error_full_zoom','Shell-balanced S0/S1'),('abs_error_global_zoom','Global loss S0/S1')]):
        ax=fig.add_subplot(b[i]);show(ax,case/(field+'.png'),title)
        if i==0:letter(ax,'b')
    ax=fig.add_subplot(gs[0,2]);
    groups=x.groupby('shell_id')
    xs=np.arange(5)
    for field,title in [('musrnet','Full'),('global_loss','Global loss'),
                        ('pure_hurdle','Pure hurdle'),('direct','Direct')]:
        err=x.assign(error=np.abs(x[field]-x.true)).groupby('shell_id')['error'].mean().reindex(xs)
        ax.plot(xs,err,marker='o',ms=3,lw=1.2,color=C[field],label=title)
    ax.set_xticks(xs,['S0','S1','S2','S3','S4'])
    ax.set_xlabel('Mutation-centred shell')
    ax.set_ylabel(r'MAE ($\mathrm{\AA}$)')
    ax2=ax.twinx()
    ax2.plot(xs,groups.log_variance.mean().reindex(xs),ls=':',color='#64748B',marker='x',label='Mean log variance')
    ax2.set_ylabel(r'Mean $\log\hat\sigma_i^2$',color='#64748B',fontsize=8.5)
    ax.legend(frameon=False,fontsize=6,loc='upper right')
    tidy(ax)
    letter(ax,'c')
    fig.savefig(OUT/'figure_mechanism_3d.pdf',dpi=300)
    fig.savefig(OUT/'figure_mechanism_3d.png',dpi=300)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--main-only',action='store_true')
    args=parser.parse_args()
    main_figure()
    if not args.main_only:mechanism_figure()


if __name__=='__main__':
    main()
