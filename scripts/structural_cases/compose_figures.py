"""Compose the structural main figure and the mechanism figure as vector PDFs.

Raster molecular renderings are embedded at full PyMOL resolution; charts,
labels, legends and statistics remain vector PDF objects.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from common import OUT

C = {'truth':'#18181B','musrnet':'#BE123C','geometry':'#0D9488',
     'coordinate':'#7C3AED','af3':'#94A3B8','esmfold2':'#D97706',
     'global_loss':'#0284C7','direct':'#64748B','pure_hurdle':'#D97706'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':8.5,'axes.titlesize':10,'axes.labelsize':10,'xtick.labelsize':8.5,
    'ytick.labelsize':8.5,'legend.fontsize':7.5,'pdf.fonttype':42,'ps.fonttype':42})


def tidy(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def show(ax, file: Path, title=''):
    ax.imshow(plt.imread(file))
    ax.axis('off')
    if title: ax.set_title(title, fontsize=8.5, pad=2)


def letter(fig, ax, label):
    p=ax.get_position()
    fig.text(p.x0-0.01,p.y1+0.015,label,weight='bold',fontsize=16,va='top')


def calc_radial(ax, df):
    shell=np.arange(5)
    values=[]
    for field in ('true','musrnet'):
        means=df.groupby('shell_id')[field].mean().reindex(shell)
        maxima=df.groupby('shell_id')[field].max().reindex(shell)
        ax.plot(shell,means,marker='o',label=field.title(),color=C[field],lw=1.4)
        if field=='true': ax.scatter(shell,maxima,color='#D97706',marker='^',s=17,label='Observed shell maximum',zorder=5)
    ax.set_xticks(shell,['S0','S1','S2','S3','S4'])
    ax.set_ylabel(r'C$\alpha$ displacement ($\mathrm{\AA}$)')
    ax.set_xlabel('Radial shell')
    tidy(ax)
    ax.legend(frameon=False,fontsize=6,loc='upper right')


def main_figure():
    case1=OUT/'cases/case1_comparison'
    cases={name:OUT/'cases'/name for name in ('case2_local','case3_nonlocal')}
    x=pd.read_csv(case1/'residues.csv')
    f=pd.read_csv(OUT/'fold_baseline_per_sample.csv',dtype={'sample_id':str})
    sample_id=pd.read_csv(OUT/'selected_cases.csv',dtype={'sample_id':str}).iloc[0].sample_id
    mets=f[f.sample_id==sample_id].set_index('model')

    fig=plt.figure(figsize=(12,9),facecolor='white')
    gs=GridSpec(3,2,figure=fig,height_ratios=[1.23,1.05,1.06],hspace=.40,wspace=.18,
                left=.055,right=.985,top=.965,bottom=.075)
    a=GridSpecFromSubplotSpec(1,2,subplot_spec=gs[0,0],wspace=.02)
    axa1=fig.add_subplot(a[0]);axa2=fig.add_subplot(a[1]);
    show(axa1,case1/'overlay.png','WT / mutant overlay')
    show(axa2,case1/'zoom.png','Local conformational shift')
    letter(fig,axa1,'a')

    b=GridSpecFromSubplotSpec(1,5,subplot_spec=gs[0,1],wspace=.02)
    names=[('true','Experimental'),('musrnet','MuSRNet'),('geometry','Geometry GNN'),
           ('coordinate','Coordinate residual'),('af3','AlphaFold3')]
    for i,(field,title) in enumerate(names):
        ax=fig.add_subplot(b[i])
        show(ax,case1/(field+'.png'),title)
        model = 'musrnet' if field == 'true' else field
        if field!='true':
            r=mets.loc[model]
            ax.text(.5,-.025,f"MAE {r.shell_mae:.2f} AUPRC {r.auprc:.2f}",
                    transform=ax.transAxes,ha='center',va='top',fontsize=6)
        if i==0:letter(fig,ax,'b')

    c=GridSpecFromSubplotSpec(2,1,subplot_spec=gs[1,0],hspace=.08,height_ratios=[1.3,.7])
    ac0=fig.add_subplot(c[0]);ac1=fig.add_subplot(c[1],sharex=ac0)
    shade={0:'#FDE68A',1:'#CCFBF1',2:'#DCEAFE'}
    for i,row in x.iterrows():
        k=int(row.shell_id)
        if k in shade:
            for ax in (ac0,ac1): ax.axvspan(i+.5,i+1.5,color=shade[k],alpha=.18,lw=0)
    for field,style in [('true','-'),('musrnet','-'),('geometry','--'),('coordinate','-.'),('af3',':')]:
        ac0.plot(x.position,x[field],style,lw=1.15,color=C[field],label=field.replace('_',' ').title())
    ac0.axhline(1,color='#888',ls=':',lw=.7)
    ac0.set_ylabel(r'$d_i$ ($\mathrm{\AA}$)')
    ac0.legend(frameon=False,ncol=3,fontsize=6,loc='upper right')
    for field in ('musrnet','geometry','coordinate'):
        ac1.plot(x.position,x[f'prob_{field}'],color=C[field],lw=1.0)
    ac1.fill_between(x.position,0,x.true_perturbed.astype(float),alpha=.12,color='#D97706')
    ac1.set_ylabel(r'$\hat p_i$')
    ac1.set_xlabel('Residue index')
    for ax in (ac0,ac1): tidy(ax)
    letter(fig,ac0,'c')

    d=GridSpecFromSubplotSpec(1,3,subplot_spec=gs[1,1],wspace=.36,width_ratios=[1,1,1])
    metrics=pd.read_csv(OUT/'fold_baseline_summary.csv')
    order=['musrnet','geometry','coordinate','esmfold2','af3']
    labels=['MuSRNet','Geometry','Coord.','ESMFold2','AF3']
    for i,metric in enumerate(['shell_mae','auprc']):
        ax=fig.add_subplot(d[i]);sub=metrics[metrics.metric==metric].set_index('model').loc[order]
        vals=sub.cluster_mean.to_numpy(float)
        ax.bar(range(5),vals,color=[C[q] for q in order],width=.72)
        ax.errorbar(range(5),vals,yerr=[vals-sub.ci95_low.to_numpy(float),sub.ci95_high.to_numpy(float)-vals],
                    fmt='none',ecolor='#222',capsize=2,lw=.7)
        ax.set_xticks(range(5),labels,rotation=55,ha='right')
        ax.set_title('Cluster shell-MAE' if i==0 else 'Cluster AUPRC',fontsize=8.5)
        tidy(ax)
        if i==0:letter(fig,ax,'d')
    ax=fig.add_subplot(d[2]);subset=f[f.model.isin(['musrnet','af3','esmfold2'])]
    for key in ('musrnet','esmfold2','af3'):
        df=subset[subset.model==key]
        ax.scatter(df.max_true_disp,df.max_pred_disp,color=C[key],s=6,alpha=.45,label=key)
    lim=float(max(subset.max_true_disp.max(),subset.max_pred_disp.quantile(.98)))
    ax.plot([0,lim],[0,lim],':',color='#888',lw=.8)
    ax.set_xlim(0,lim);ax.set_ylim(0,lim)
    ax.set_xlabel(r'Observed max ($\mathrm{\AA}$)',fontsize=8)
    ax.set_ylabel(r'Predicted max ($\mathrm{\AA}$)',fontsize=8)
    ax.set_title('Response amplitude',fontsize=8.5)
    ax.legend(frameon=False,fontsize=6);tidy(ax)

    for col,case,label in [(0,'case2_local','e'),(1,'case3_nonlocal','f')]:
        inner=GridSpecFromSubplotSpec(1,2,subplot_spec=gs[2,col],width_ratios=[1.15,1],wspace=.08)
        aimg=fig.add_subplot(inner[0]);arad=fig.add_subplot(inner[1]);
        p=cases[case];show(aimg,p/'radial.png','Local response' if col==0 else 'Non-local response')
        calc_radial(arad,pd.read_csv(p/'residues.csv'))
        letter(fig,aimg,label)
    fig.savefig(OUT/'figure_structural_cases.pdf',dpi=300)
    fig.savefig(OUT/'figure_structural_cases.png',dpi=300)
    plt.close(fig)


def mechanism_figure():
    case=OUT/'cases/case4_mechanism'
    x=pd.read_csv(case/'residues.csv')
    fig=plt.figure(figsize=(11,4.5),facecolor='white')
    gs=GridSpec(1,3,figure=fig,left=.035,right=.975,bottom=.19,top=.90,wspace=.20,width_ratios=[1.34,1.0,1.16])
    a=GridSpecFromSubplotSpec(1,4,subplot_spec=gs[0,0],wspace=.015)
    for i,(field,title) in enumerate([('background','Background'),('gated_excess','Gated excess'),
                                      ('musrnet','Combined'),('true','Observed')]):
        ax=fig.add_subplot(a[i]);show(ax,case/(field+'.png'),title)
        if i==0:letter(fig,ax,'a')
    b=GridSpecFromSubplotSpec(1,2,subplot_spec=gs[0,1],wspace=.02)
    for i,(field,title) in enumerate([('abs_error_full_zoom','Shell-balanced S0/S1'),('abs_error_global_zoom','Global loss S0/S1')]):
        ax=fig.add_subplot(b[i]);show(ax,case/(field+'.png'),title)
        if i==0:letter(fig,ax,'b')
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
    letter(fig,ax,'c')
    fig.savefig(OUT/'figure_mechanism_3d.pdf',dpi=300)
    fig.savefig(OUT/'figure_mechanism_3d.png',dpi=300)
    plt.close(fig)


if __name__=='__main__':
    main_figure()
    mechanism_figure()
    print('Wrote two vector-PDF composite figures')
