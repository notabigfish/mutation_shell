from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path("compact_results")
OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def csv(path):
    return pd.read_csv(ROOT / path)


def js(path):
    with open(ROOT / path) as f:
        return json.load(f)


def panel(ax, label):
    ax.text(
        -0.16, 1.06, label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# Figure 1: primary_and_robustness.pdf
# ============================================================

def load_primary_summary():
    d = csv("strict_baseline_comparison_base_v5_seed42/summary_metrics.csv")
    order = [
        "zero_response", "global_mean", "shell_mean",
        "mutation_type_shell_mean", "esm_mlp",
        "geometry_gnn", "coordinate_residual", "base_v5_seed42",
    ]
    return d.set_index("model_name").loc[order].reset_index()


def paired_primary(metric):
    d = csv("strict_baseline_comparison_base_v5_seed42/statistical_tests.csv")
    return d[
        (d["metric"] == metric)
        & d["baseline"].isin(["geometry_gnn", "coordinate_residual"])
    ].copy()

def plot_paired_primary(ax, metric, xlabel, title):
    d = paired_primary(metric)
    baselines = ["geometry_gnn", "coordinate_residual"]
    labels = ["Geometry GNN", "Coordinate residual"]

    for i, baseline in enumerate(baselines):
        r = d.loc[d["baseline"] == baseline].iloc[0]
        mean = float(r["mean_diff"])
        low = float(r["bootstrap_95ci_low"])
        high = float(r["bootstrap_95ci_high"])

        ax.errorbar(
            mean, i,
            xerr=[[mean - low], [high - mean]],
            fmt="o", capsize=3,
        )

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    clean(ax)

def plot_primary_and_robustness():
    summary = load_primary_summary()

    name_map = {
        "zero_response": "Zero response",
        "global_mean": "Global mean",
        "shell_mean": "Shell mean",
        "mutation_type_shell_mean": "Mutation-type\nshell mean",
        "esm_mlp": "ESM-only MLP",
        "geometry_gnn": "Geometry GNN",
        "coordinate_residual": "Coordinate residual",
        "base_v5_seed42": "MuSRNet",
    }

    summary["label"] = summary["model_name"].map(name_map)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # a: cluster-average shell-MAE
    y = np.arange(len(summary))
    bars = ax_a.barh(y, summary["cluster_avg_shell_mae"])

    for i, bar in enumerate(bars):
        if summary.iloc[i]["model_name"].startswith("base_v5"):
            bar.set_hatch("//")

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(summary["label"])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Cluster-average shell-MAE (Å)")
    ax_a.set_title("Displacement prediction")
    clean(ax_a)
    panel(ax_a, "a")

    # b: cluster-average AUPRC
    bars = ax_b.barh(y, summary["cluster_avg_auprc"])

    for i, bar in enumerate(bars):
        if summary.iloc[i]["model_name"].startswith("base_v5"):
            bar.set_hatch("//")

    ax_b.set_yticks(y)
    ax_b.set_yticklabels([])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Cluster-average AUPRC")
    ax_b.set_title("Perturbed-residue localization")
    clean(ax_b)
    panel(ax_b, "b")

    # c: paired shell-MAE differences
    plot_paired_primary(
        ax_c, "shell_mae",
        r"$\Delta$ cluster shell-MAE (Å)",
        "Paired cluster displacement difference",
    )
    panel(ax_c, "c")

    # d: paired AUPRC differences
    plot_paired_primary(
        ax_d, "perturbed_auprc",
        r"$\Delta$ cluster AUPRC",
        "Paired cluster localization difference",
    )
    panel(ax_d, "d")

    fig.tight_layout()
    fig.savefig(
        OUT / "primary_and_robustness.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# Figure 2: sensitivity_and_generalization.pdf
# ============================================================

def alignment_cluster_metrics():
    d = csv(
        "alignment_sensitivity/"
        "cluster_compare_ref_kabsch_all/"
        "cluster_metrics_wide.csv"
    )

    return (
        d.groupby("alignment_variant", as_index=False)
        .agg(
            cluster_shell_mae=("shell_mae_cluster_mean", "mean"),
            cluster_auprc=("perturbed_auprc_cluster_mean", "mean"),
        )
    )


def plot_sensitivity_and_generalization():
    label4 = js(
        "alignment_sensitivity/"
        "label_compare_kabsch_exclude_4A_vs_all/summary.json"
    )
    label8 = js(
        "alignment_sensitivity/"
        "label_compare_kabsch_exclude_8A_vs_all/summary.json"
    )

    align = alignment_cluster_metrics()

    variants = [
        "kabsch_all",
        "kabsch_exclude_4A",
        "kabsch_exclude_8A",
    ]
    variant_labels = [
        "All residues",
        "Exclude 4 Å",
        "Exclude 8 Å",
    ]

    align = (
        align.set_index("alignment_variant")
        .loc[variants]
        .reset_index()
    )

    time = csv("time_split_comparison/summary_metrics.csv")
    time_order = [
        "base_v5_time",
        "geometry_gnn_time",
        "shell_mean_time",
        "mutation_type_shell_mean_time",
    ]
    time_labels = [
        "MuSRNet",
        "Geometry GNN",
        "Shell mean",
        "Mutation-type\nshell mean",
    ]
    time = (
        time.set_index("model_name")
        .loc[time_order]
        .reset_index()
    )

    valid_thr = csv("threshold_sensitivity_valid.csv")

    test_thr = csv("threshold_sensitivity.csv")
    test_thr = test_thr[
        (test_thr["model_name"] == "base_v5_seed42")
        & (test_thr["radius_threshold"] == 8.0)
        & (test_thr["displacement_threshold"] == 1.0)
    ].sort_values("response_threshold")

    fig, axes = plt.subplots(2, 4, figsize=(10, 8))

    # a: displacement label difference
    ax = axes[0, 0]
    xlabels = ["Exclude 4 Å", "Exclude 8 Å"]
    vals = [
        label4["mean_abs_diff_disp"],
        label8["mean_abs_diff_disp"],
    ]
    ax.bar(np.arange(2), vals)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(xlabels, rotation=20, ha="right")
    ax.set_ylabel("Mean absolute label difference (Å)")
    ax.set_title("Alignment-label deviation")
    clean(ax)
    panel(ax, "a")

    # b: perturbed-label agreement
    ax = axes[0, 1]
    vals = [
        label4["perturbed_jaccard"],
        label8["perturbed_jaccard"],
    ]
    ax.bar(np.arange(2), vals)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(xlabels, rotation=20, ha="right")
    ax.set_ylim(0.94, 1.0)
    ax.set_ylabel("Perturbed-label Jaccard")
    ax.set_title("Perturbed-label agreement")

    for i, (v, info) in enumerate(
        zip(vals, [label4, label8])
    ):
        ax.text(
            i,
            v - 0.002,
            f"class disagreement\n"
            f"{100 * info['class_disagreement_fraction']:.2f}%",
            ha="center",
            va="top",
            fontsize=6,
        )

    clean(ax)
    panel(ax, "b")

    # c: alignment shell-MAE
    ax = axes[0, 2]
    ax.bar(
        np.arange(3),
        align["cluster_shell_mae"],
    )
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(
        variant_labels,
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Cluster shell-MAE (Å)")
    ax.set_title("Alignment sensitivity")
    clean(ax)
    panel(ax, "c")

    # d: alignment AUPRC
    ax = axes[0, 3]
    ax.bar(
        np.arange(3),
        align["cluster_auprc"],
    )
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(
        variant_labels,
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Cluster AUPRC")
    ax.set_title("Alignment sensitivity")
    clean(ax)
    panel(ax, "d")

    # e: latest-release shell-MAE
    ax = axes[1, 0]
    ax.bar(
        np.arange(len(time)),
        time["cluster_avg_shell_mae"],
    )
    ax.set_xticks(np.arange(len(time)))
    ax.set_xticklabels(
        time_labels,
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Cluster shell-MAE (Å)")
    ax.set_title("Latest-release holdout")
    clean(ax)
    panel(ax, "e")

    # f: latest-release AUPRC
    ax = axes[1, 1]
    ax.bar(
        np.arange(len(time)),
        time["cluster_avg_auprc"],
    )
    ax.set_xticks(np.arange(len(time)))
    ax.set_xticklabels(
        time_labels,
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Cluster AUPRC")
    ax.set_title("Latest-release holdout")
    clean(ax)
    panel(ax, "f")

    # g: threshold sensitivity — radius
    ax = axes[1, 2]
    ax.plot(
        valid_thr["response_threshold"],
        valid_thr["cluster_avg_radius_mae"],
        marker="o",
        label="Validation",
    )
    ax.plot(
        test_thr["response_threshold"],
        test_thr["cluster_avg_radius_mae"],
        marker="o",
        label="Test",
    )
    ax.set_xlabel("Perturbed-probability threshold")
    ax.set_ylabel("Cluster radius MAE (Å)")
    ax.set_title("Derived radius")
    ax.legend(frameon=False)
    clean(ax)
    panel(ax, "g")

    # h: threshold sensitivity — class
    ax = axes[1, 3]
    ax.plot(
        valid_thr["response_threshold"],
        valid_thr["cluster_avg_class_macro_f1"],
        marker="o",
        label="Validation",
    )
    ax.plot(
        test_thr["response_threshold"],
        test_thr["cluster_avg_class_macro_f1"],
        marker="o",
        label="Test",
    )
    ax.set_xlabel("Perturbed-probability threshold")
    ax.set_ylabel("Cluster macro-F1")
    ax.set_title("Derived response class")
    clean(ax)
    panel(ax, "h")

    fig.tight_layout()
    fig.savefig(
        OUT / "sensitivity_and_generalization.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

# ============================================================
# Figure 3: counterfactual_and_stratification.pdf
# ============================================================

def load_counterfactual():
    frames = []

    for seed in range(10):
        p = (
            ROOT
            / "base_v5_seed42"
            / f"counterfactual_corrected_seed{seed}"
            / "counterfactual_summary.csv"
        )
        d = pd.read_csv(p)
        d["counterfactual_seed"] = seed
        frames.append(d)

    return pd.concat(frames, ignore_index=True)


def collect_stratified(metric):
    selected = [
        ("exposure_group", "buried", "Exposure: buried"),
        ("exposure_group", "exposed", "Exposure: exposed"),
        ("secondary_structure_group", "helix", "Secondary: helix"),
        ("secondary_structure_group", "sheet", "Secondary: sheet"),
        ("secondary_structure_group", "loop", "Secondary: loop"),
        ("length_group", "short", "Length: short"),
        ("length_group", "long", "Length: long"),
        ("resolution_group", "high_resolution", "Resolution: high"),
        ("resolution_group", "low_resolution", "Resolution: low"),
        (
            "substitution_size_group",
            "small_to_large",
            "Substitution: small→large",
        ),
        (
            "charge_group",
            "neutral_to_charged",
            "Charge: neutral→charged",
        ),
    ]

    comparator_dirs = {
        "Geometry GNN": "geometry_gnn",
        "Coordinate residual": "coordinate_residual",
    }

    rows = []

    for comparator_label, comparator_dir in comparator_dirs.items():
        d = csv(
            "biological_stratification/"
            f"v5_seed42_vs_{comparator_dir}/"
            "pairwise_stratified_diff.csv"
        )

        for factor, group, label in selected:
            r = d[
                (d["factor"] == factor)
                & (d["group"] == group)
                & (d["metric"] == metric)
            ]

            if len(r) != 1:
                raise ValueError(f"Missing stratum: {factor}/{group}/{metric}")

            r = r.iloc[0]
            mean = float(r["mean_diff"])
            low = float(r["bootstrap_ci_low"])
            high = float(r["bootstrap_ci_high"])

            if metric == "shell_mae":
                mean, low, high = -mean, -high, -low

            rows.append({
                "comparator": comparator_label,
                "stratum": label,
                "mean": mean,
                "low": low,
                "high": high,
            })

    return pd.DataFrame(rows)


def plot_seed_points(ax, data, variants, labels, metric, ylabel):
    for i, variant in enumerate(variants):
        values = (
            data.loc[data["variant"] == variant, metric]
            .dropna()
            .to_numpy()
        )

        offsets = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(
            np.full(len(values), i) + offsets,
            values,
            s=14,
            alpha=0.7,
        )

        ax.errorbar(
            i,
            values.mean(),
            yerr=values.std(ddof=1) if len(values) > 1 else 0,
            fmt="D",
            capsize=3,
        )

    ax.axhline(0, linewidth=0.8, linestyle="--")
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    clean(ax)


def plot_stratified_advantage(ax, metric, xlabel):
    d = collect_stratified(metric)
    order = list(dict.fromkeys(d["stratum"]))
    y = np.arange(len(order))

    offsets = {
        "Geometry GNN": -0.12,
        "Coordinate residual": 0.12,
    }

    for comparator, offset in offsets.items():
        x = (
            d[d["comparator"] == comparator]
            .set_index("stratum")
            .loc[order]
        )

        means = x["mean"].to_numpy(dtype=float)
        lows = x["low"].to_numpy(dtype=float)
        highs = x["high"].to_numpy(dtype=float)

        ax.errorbar(
            means, y + offset,
            xerr=np.vstack([means - lows, highs - means]),
            fmt="o", capsize=2, label=comparator,
        )

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    clean(ax)


def plot_counterfactual_and_stratification():
    cf = load_counterfactual()

    variants = [
        "shuffle_site",
        "shuffle_mutant_aa",
        "reverse_mutation",
        "remove_mutation_context",
    ]

    variant_labels = [
        "Shuffle site",
        "Shuffle mutant AA",
        "Reverse mutation",
        "Remove context",
    ]

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.0, 0.75, 2.2],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    # a: counterfactual shell-MAE change
    plot_seed_points(
        ax_a,
        cf,
        variants,
        variant_labels,
        "delta_shell_mae",
        r"$\Delta$ shell-MAE (Å)",
    )
    ax_a.set_title("Counterfactual displacement effect")
    panel(ax_a, "a")

    # b: counterfactual AUPRC change
    plot_seed_points(
        ax_b,
        cf,
        variants,
        variant_labels,
        "delta_auprc",
        r"$\Delta$ AUPRC",
    )
    ax_b.set_title("Counterfactual localization effect")
    panel(ax_b, "b")

    # c: WT-WT negative input
    original = (
        cf[cf["variant"] == "original"]
        .set_index("counterfactual_seed")
    )
    wtwt = (
        cf[cf["variant"] == "wt_wt_negative"]
        .set_index("counterfactual_seed")
    )

    ratios = {
        "Mean predicted\ndisplacement":
            (
                wtwt["mean_pred_displacement"]
                / original["mean_pred_displacement"]
            ).to_numpy(),
        "Mean perturbed\nprobability":
            (
                wtwt["mean_pred_perturbed_prob"]
                / original["mean_pred_perturbed_prob"]
            ).to_numpy(),
    }

    for i, (label, values) in enumerate(ratios.items()):
        offsets = np.linspace(-0.08, 0.08, len(values))

        ax_c.scatter(
            np.full(len(values), i) + offsets,
            values,
            s=16,
            alpha=0.7,
        )

        ax_c.errorbar(
            i,
            values.mean(),
            yerr=values.std(ddof=1),
            fmt="D",
            capsize=3,
        )

    ax_c.axhline(
        1.0,
        linewidth=0.8,
        linestyle="--",
    )
    ax_c.set_xticks(np.arange(len(ratios)))
    ax_c.set_xticklabels(list(ratios))
    ax_c.set_ylabel("WT–WT / original prediction")
    ax_c.set_title("Response retained by the WT–WT negative input")
    clean(ax_c)
    panel(ax_c, "c")

    # d: stratified shell-MAE advantage
    plot_stratified_advantage(
        ax_d,
        metric="shell_mae",
        xlabel=(
            "MuSRNet advantage in shell-MAE (Å)\n"
            "(positive favours MuSRNet)"
        ),
    )
    ax_d.set_title("Displacement across structural strata")
    ax_d.legend(frameon=False)
    panel(ax_d, "d")

    # e: stratified AUPRC advantage
    plot_stratified_advantage(
        ax_e,
        metric="perturbed_auprc",
        xlabel=(
            "MuSRNet advantage in AUPRC\n"
            "(positive favours MuSRNet)"
        ),
    )
    ax_e.set_title("Localization across structural strata")
    ax_e.set_yticklabels([])
    panel(ax_e, "e")

    fig.tight_layout()

    fig.savefig(
        OUT / "counterfactual_and_stratification.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":
    plot_primary_and_robustness()
    plot_sensitivity_and_generalization()
    plot_counterfactual_and_stratification()