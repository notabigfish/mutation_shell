"""Paired 221-cluster comparisons with annotated effect-size figure."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from common import OUT, ci95

COMPARATORS = ('af3', 'esmfold2')
PRIMARY = ('shell_mae', 'auprc')
METRICS = ('shell_mae', 'auprc', 'pearson', 'spearman')


def holm(p):
    p=np.asarray(p,dtype=float)
    order=np.argsort(p)
    adjusted=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order])
    result=np.empty(len(p));result[order]=np.minimum(1,adjusted)
    return result


def compute():
    data=pd.read_csv(OUT/'fold_baseline_per_sample.csv',
                     dtype={'sample_id':str,'cluster_id_30':str})
    assert data.cluster_id_30.nunique()==221
    assert not data.duplicated(['model','sample_id']).any()
    musr=data[data.model=='musrnet']
    assert len(musr)==221 and musr.cluster_id_30.nunique()==221
    summary=[];detailed=[]
    for other in COMPARATORS:
        paired=musr.merge(data[data.model==other],on=['sample_id','cluster_id_30'],
                          suffixes=('_musr','_other'),validate='one_to_one')
        assert len(paired)==221, f'Samples differ for {other}'
        for metric in METRICS:
            m=paired[f'{metric}_musr'].to_numpy(float)
            o=paired[f'{metric}_other'].to_numpy(float)
            valid=np.isfinite(m)&np.isfinite(o)
            dif=(o-m if metric=='shell_mae' else m-o)[valid]
            low,high=ci95(dif)
            pvalue=float(wilcoxon(dif).pvalue) if np.any(dif) else 1.0
            summary.append({'opponent':other,'metric':metric,'n_paired_clusters':int(valid.sum()),
                            'mean_musrnet':float(m[valid].mean()),
                            'mean_opponent':float(o[valid].mean()),
                            'paired_mean_difference':float(dif.mean()),
                            'paired_95ci_low':low,'paired_95ci_high':high,
                            'wilcoxon_p':pvalue})
            detailed.extend({'opponent':other,'metric':metric,
                             'cluster_id_30':str(k),'paired_difference':float(v)}
                            for k,v in zip(paired.cluster_id_30[valid],dif))
    result=pd.DataFrame(summary)
    mask=result.metric.isin(PRIMARY)
    result.loc[mask,'holm_p_primary_four_tests']=holm(result.loc[mask,'wilcoxon_p'].to_numpy())
    result.to_csv(OUT/'fold_paired_cluster_summary.csv',index=False)
    pd.DataFrame(detailed).to_csv(OUT/'fold_paired_cluster_differences.csv',index=False)
    print(result[result.metric.isin(PRIMARY)].to_string(index=False))
    return result


def plot(result):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                         'font.size':10,'axes.labelsize':11,'pdf.fonttype':42})
    fig=plt.figure(figsize=(13.5,5.0),facecolor='white')
    gs=fig.add_gridspec(2,2,height_ratios=[1,.60],left=.08,right=.985,top=.90,
                        bottom=.08,wspace=.31,hspace=.38)
    for column,metric in enumerate(PRIMARY):
        ax=fig.add_subplot(gs[0,column]);info=fig.add_subplot(gs[1,column]);info.axis('off')
        rows=result[result.metric==metric].set_index('opponent')
        for y,model in enumerate(COMPARATORS):
            r=rows.loc[model]; val=r.paired_mean_difference
            ax.errorbar(val,y,xerr=[[val-r.paired_95ci_low],[r.paired_95ci_high-val]],
                        fmt='o',markersize=7,capsize=4,lw=1.7,
                        color=('#BE123C' if model=='af3' else '#0D9488'))
            label='AlphaFold3' if model=='af3' else 'ESMFold2'
            info.text(.02,.84-.54*y,
                      f"{label}: n={int(r.n_paired_clusters)};  Δ={val:+.3f}; "
                      f"95% CI [{r.paired_95ci_low:+.3f}, {r.paired_95ci_high:+.3f}]",
                      fontsize=10,ha='left',va='top')
            info.text(.02,.64-.54*y,
                      f"Wilcoxon P={r.wilcoxon_p:.3g}; Holm-adjusted P={r.holm_p_primary_four_tests:.3g}",
                      fontsize=9,color='#555',ha='left',va='top')
        ax.axvline(0,color='#777',ls='--',lw=.9)
        ax.set_yticks((0,1),('AlphaFold3','ESMFold2'))
        ax.set_ylim(1.45,-.45)
        if metric=='shell_mae':
            ax.set_title('Shell-MAE advantage (Å)',fontsize=13)
            ax.set_xlabel('Δ = Folding model − MuSRNet (Å)')
        else:
            ax.set_title('AUPRC advantage',fontsize=13)
            ax.set_xlabel('Δ = MuSRNet − Folding model')
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'figure_panel_d_paired.pdf',dpi=300)
    fig.savefig(OUT/'figure_panel_d_paired.png',dpi=300)
    plt.close(fig)


def main():
    result=compute();plot(result)


if __name__=='__main__':
    main()
