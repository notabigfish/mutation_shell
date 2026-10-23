"""Sample-level P@5%, equal-cluster comparison, bootstrap CI and Wilcoxon test."""
from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from common import OUT, ci95


def main():
    a = pd.read_csv(OUT / 'case_sample_metrics_musrnet.csv', dtype={'sample_id': str, 'cluster_id_30': str})
    b = pd.read_csv(OUT / 'case_sample_metrics_geometry.csv', dtype={'sample_id': str, 'cluster_id_30': str})
    a = a[['sample_id', 'cluster_id_30', 'precision_at5']].rename(columns={'precision_at5': 'musrnet'})
    b = b[['sample_id', 'precision_at5']].rename(columns={'precision_at5': 'geometry'})
    joined = a.merge(b, on='sample_id', validate='one_to_one')
    joined['difference'] = joined.musrnet - joined.geometry
    joined.to_csv(OUT / 'precision_at5_sample.csv', index=False)
    cluster = joined.groupby('cluster_id_30', as_index=False)[['musrnet', 'geometry', 'difference']].mean()
    cluster.to_csv(OUT / 'precision_at5_cluster.csv', index=False)
    diff = cluster.difference.to_numpy(float)
    low, high = ci95(diff)
    summary = pd.DataFrame([{
        'n_samples': len(joined), 'n_clusters': len(cluster),
        'musrnet_cluster_mean': cluster.musrnet.mean(),
        'geometry_cluster_mean': cluster.geometry.mean(),
        'mean_paired_difference': diff.mean(),
        'bootstrap_95ci_low': low, 'bootstrap_95ci_high': high,
        'paired_wilcoxon_p': wilcoxon(diff).pvalue,
    }])
    summary.to_csv(OUT / 'precision_at5_summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
