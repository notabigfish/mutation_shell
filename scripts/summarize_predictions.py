import json
from pathlib import Path
import pandas as pd

FILES = {
    "cluster_test": "outputs/call/base_v5_seed42/predictions_test.csv",
    "cluster_valid": "outputs/call/base_v5_seed42/predictions_valid.csv",
    "time_test": "outputs/call/base_v5_time/predictions_test.csv",
    "time_valid": "outputs/call/base_v5_time/predictions_valid.csv",
}

out = {}

for name, path in FILES.items():
    path = Path(path)
    if not path.exists():
        continue

    n = 0
    perturbed = 0.0
    disp_sum = 0.0
    shell_counts = {}
    sample_ids = set()
    cluster_ids = set()

    usecols = [
        "sample_id",
        "cluster_id_30",
        "shell_id",
        "true_displacement",
        "true_perturbed",
    ]

    for df in pd.read_csv(path, usecols=usecols, chunksize=1_000_000):
        n += len(df)
        perturbed += df["true_perturbed"].sum()
        disp_sum += df["true_displacement"].sum()

        sample_ids.update(df["sample_id"].astype(str).unique())
        cluster_ids.update(df["cluster_id_30"].astype(str).unique())

        vc = df["shell_id"].value_counts()
        for k, v in vc.items():
            shell_counts[str(int(k))] = shell_counts.get(str(int(k)), 0) + int(v)

    out[name] = {
        "n_residues": n,
        "n_samples": len(sample_ids),
        "n_clusters": len(cluster_ids),
        "perturbed_fraction": perturbed / n,
        "mean_true_displacement": disp_sum / n,
        "shell_residue_counts": shell_counts,
    }

Path("compact_results/dataset_prediction_summary.json").write_text(
    json.dumps(out, indent=2),
    encoding="utf-8",
)
