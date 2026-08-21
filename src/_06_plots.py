"""
Figures for all three approaches, any property, plus the cross-approach comparison.
Reads the raw run CSVs (via _05_analysis's loaders) and the metrics CSVs _05_analysis
writes -- run that first.

Usage:
    python _06_plots.py --approach 1 --property esol
    python _06_plots.py --approach 2 --property jnk3
    python _06_plots.py --approach 3 --property esol
    python _06_plots.py --approach compare --property jnk3
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
from _01_seed_molecules import get_seeds  # noqa: E402
from _05_analysis import (  # noqa: E402
    load_runs_1, load_runs_2, load_runs_3, _results_dir, _is_valid,
    _mean_pairwise_similarity, RESULTS_ROOT,
)

PLOTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "plots")
VARIANT_COLORS = {"A": "#d62728", "B": "#1f77b4", "C": "#2ca02c"}
ACQ_COLORS = {"ei": "#1f77b4", "ucb": "#d62728"}


def _plots_dir(approach, property_name):
    base = os.path.join(PLOTS_ROOT, f"approach{approach}")
    return base if property_name == "esol" else os.path.join(base, property_name)


def _save(plots_dir, name):
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _baseline_summary(property_name):
    name = "baseline_summary.csv" if property_name == "esol" else f"baseline_summary_{property_name}.csv"
    path = os.path.join(RESULTS_ROOT, name)
    return pd.read_csv(path).iloc[0] if os.path.exists(path) else None


# --------------------------------------------------------------------------- #
# Approach 1
# --------------------------------------------------------------------------- #

def plots_approach1(property_name):
    runs = load_runs_1(property_name)
    if not runs:
        print("No Approach 1 result CSVs found.")
        return
    prop = C.PROPERTIES[property_name]
    start = max(m["score"] for m in get_seeds(property_name))
    d = _plots_dir(1, property_name)

    plt.figure(figsize=(8, 5))
    for ordinal, ((variant, run), df) in enumerate(sorted(runs.items())):
        iters = [0] + df["iteration"].tolist()
        best = [start] + df["best_so_far"].astype(float).tolist()
        plt.plot(iters, best, marker="o", markersize=3, label=f"Variant {variant} run {run}")
    plt.axhline(start, ls="--", lw=1, color="gray", label=f"seed best ({start:+.2f})")
    plt.xlabel("Iteration")
    plt.ylabel(prop["label"])
    plt.title("Approach 1: convergence of best-so-far")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save(d, "convergence.png")

    by_variant_invalid = {}
    by_variant_div = {}
    for (variant, _), df in runs.items():
        valid = _is_valid(df["valid"])
        by_variant_invalid.setdefault(variant, []).append(1 - valid.mean())
        smiles = df.loc[valid, "canonical_smiles"].dropna().tolist()
        sim = _mean_pairwise_similarity(smiles)
        if not np.isnan(sim):
            by_variant_div.setdefault(variant, []).append(sim)

    variants = sorted(by_variant_invalid)
    rates = [np.mean(by_variant_invalid[v]) for v in variants]
    plt.figure(figsize=(6, 4))
    plt.bar(variants, rates, color=[VARIANT_COLORS[v] for v in variants])
    for i, r in enumerate(rates):
        plt.text(i, r + 0.005, f"{r:.0%}", ha="center", fontsize=9)
    plt.ylabel("Invalid / failed proposal rate")
    plt.title("Invalid-SMILES (failed) rate per variant")
    plt.ylim(0, max(rates + [0.1]) * 1.3)
    _save(d, "invalid_rate.png")

    variants = sorted(by_variant_div)
    sims = [np.mean(by_variant_div[v]) for v in variants]
    plt.figure(figsize=(6, 4))
    plt.bar(variants, sims, color=[VARIANT_COLORS[v] for v in variants])
    for i, s in enumerate(sims):
        plt.text(i, s + 0.01, f"{s:.2f}", ha="center", fontsize=9)
    plt.ylabel("Mean pairwise Tanimoto similarity")
    plt.title("Proposal similarity per variant (lower = more diverse)")
    plt.ylim(0, 1)
    _save(d, "diversity.png")

    labels, values, colors = [], [], []
    for variant in sorted({v for v, _ in runs}):
        best = max(float(df["best_so_far"].astype(float).max())
                   for (v, _), df in runs.items() if v == variant)
        labels.append(f"Variant {variant}")
        values.append(best)
        colors.append(VARIANT_COLORS[variant])
    b = _baseline_summary(property_name)
    if b is not None:
        labels += ["Random\nbaseline", "Pool\nceiling"]
        values += [float(b["random_mean"]), float(b["ceiling"])]
        colors += ["#7f7f7f", "#000000"]
    labels.append("Seed best")
    values.append(start)
    colors.append("#bbbbbb")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:+.2f}", ha="center", fontsize=8)
    plt.ylabel(f"Best {prop['label']} found")
    plt.title("Approach 1: optimizer vs baseline vs ceiling")
    plt.grid(axis="y", alpha=0.3)
    _save(d, "comparison.png")


# --------------------------------------------------------------------------- #
# Approach 2 (generative)
# --------------------------------------------------------------------------- #

def plots_approach2(property_name):
    runs = load_runs_2(property_name)
    if not runs:
        print("No Approach 2 generative result CSVs found.")
        return
    prop = C.PROPERTIES[property_name]
    start = max(m["score"] for m in get_seeds(property_name))
    b = _baseline_summary(property_name)
    random_baseline = float(b["random_mean"]) if b is not None else None
    d = _plots_dir(2, property_name)

    plt.figure(figsize=(8, 5))
    for (acq, run), df in sorted(runs.items()):
        x = [0] + df["eval_idx"].astype(int).tolist()
        y = [start] + df["best_so_far"].astype(float).tolist()
        plt.plot(x, y, marker="o", markersize=3, alpha=0.85,
                 color=ACQ_COLORS.get(acq, "#7f7f7f"), label=f"{acq.upper()} run {run}")
    plt.axhline(start, ls="--", lw=1, color="gray", label=f"seed best ({start:+.2f})")
    if random_baseline is not None:
        plt.axhline(random_baseline, ls=":", lw=1, color="green",
                    label=f"random baseline ({random_baseline:+.2f})")
    plt.xlabel("Evaluation")
    plt.ylabel(prop["label"])
    plt.title("Approach 2 generative: best-so-far climbs above seed")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save(d, "convergence2_gen.png")

    plt.figure(figsize=(8, 5))
    for (acq, run), df in sorted(runs.items()):
        r, v = C.top_k_so_far_by_round(df["round"], df["true_score"], k=3)
        plt.plot(r, v, marker="s", markersize=5, alpha=0.85,
                 color=ACQ_COLORS.get(acq, "#7f7f7f"), label=f"{acq.upper()} run {run}")
    plt.axhline(start, ls="--", lw=1, color="gray", label=f"seed ({start:+.2f})")
    plt.xlabel("Round")
    plt.ylabel(f"Mean {prop['label']} of top-3 found so far")
    plt.title("Approach 2 generative: top-3-so-far vs round")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save(d, "topk2_gen.png")

    pred, true, acqs = [], [], []
    for (a, _), df in runs.items():
        dd = df.dropna(subset=["surrogate_label"])
        pred.extend(dd["surrogate_label"].astype(float).tolist())
        true.extend(dd["true_score"].astype(float).tolist())
        acqs.extend([a] * len(dd))
    pred, true, acqs = np.array(pred), np.array(true), np.array(acqs)
    plt.figure(figsize=(6, 6))
    for a in ("ei", "ucb"):
        m = acqs == a
        if m.any():
            plt.scatter(true[m], pred[m], s=35, alpha=0.7, color=ACQ_COLORS[a], label=f"{a.upper()} picks")
    lo = min(pred.min(), true.min()) - 0.5
    hi = max(pred.max(), true.max()) + 0.5
    plt.plot([lo, hi], [lo, hi], ls="--", color="black", label="perfect (y = x)")
    plt.xlabel(f"True {prop['label']}")
    plt.ylabel(f"LLM predicted {prop['label']}")
    plt.title("Approach 2 generative: LLM prediction vs truth")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    _save(d, "calibration2_gen.png")

    metrics_path = os.path.join(_results_dir(2, property_name), "metrics2_gen.csv")
    m = pd.read_csv(metrics_path)
    labels = [f"{r.acq.upper()}\nrun {r.run}" for r in m.itertuples()]
    vals = m["pairwise_acc"].tolist()
    colors = [ACQ_COLORS.get(r.acq, "#7f7f7f") for r in m.itertuples()]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, vals, color=colors)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    plt.axhline(0.5, ls="--", color="gray", label="coin flip (0.50)")
    plt.ylim(0, 1)
    plt.ylabel(f"Pairwise ranking accuracy vs {prop['label']}")
    plt.title("Approach 2 generative: is the LLM regressor a good ranker?")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    _save(d, "pairwise2_gen.png")


# --------------------------------------------------------------------------- #
# Approach 3 (generative)
# --------------------------------------------------------------------------- #

def plots_approach3(property_name):
    runs = load_runs_3(property_name)
    if not runs:
        print("No Approach 3 generative result CSVs found.")
        return
    prop = C.PROPERTIES[property_name]
    start = max(m["score"] for m in get_seeds(property_name))
    b = _baseline_summary(property_name)
    random_baseline = float(b["random_mean"]) if b is not None else None
    d = _plots_dir(3, property_name)
    run_colors = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}

    plt.figure(figsize=(8, 5))
    for run, (opt, _) in sorted(runs.items()):
        x = [0] + opt["eval_idx"].astype(int).tolist()
        y = [start] + opt["best_so_far"].astype(float).tolist()
        plt.plot(x, y, marker="o", markersize=3, color=run_colors.get(run, "#7f7f7f"), label=f"run {run}")
    plt.axhline(start, ls="--", lw=1, color="gray", label=f"seed best ({start:+.2f})")
    if random_baseline is not None:
        plt.axhline(random_baseline, ls=":", lw=1, color="green",
                    label=f"random baseline ({random_baseline:+.2f})")
    plt.xlabel("Evaluation")
    plt.ylabel(prop["label"])
    plt.title("Approach 3 generative: best-so-far climbs above seed")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save(d, "convergence3_gen.png")

    plt.figure(figsize=(8, 5))
    for run, (opt, _) in sorted(runs.items()):
        r, v = C.top_k_so_far_by_round(opt["round"], opt["true_score"], k=3)
        plt.plot(r, v, marker="s", markersize=5, color=run_colors.get(run, "#7f7f7f"), label=f"run {run}")
    plt.axhline(start, ls="--", lw=1, color="gray", label=f"seed ({start:+.2f})")
    plt.xlabel("Round")
    plt.ylabel(f"Mean {prop['label']} of top-3 found so far")
    plt.title("Approach 3 generative: top-3-so-far vs round")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save(d, "topk3_gen.png")

    plt.figure(figsize=(6, 6))
    for run, (_, rank) in sorted(runs.items()):
        if rank is None:
            continue
        plt.scatter(rank["true_score"], rank["bt_utility"], s=35, alpha=0.7,
                    color=run_colors.get(run, "#7f7f7f"), label=f"run {run}")
    plt.xlabel(f"True {prop['label']}")
    plt.ylabel("Bradley-Terry utility (ranker strength)")
    plt.title("Approach 3 generative: ranker utility vs truth")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save(d, "ranking_quality3_gen.png")

    metrics_path = os.path.join(_results_dir(3, property_name), "metrics3_gen.csv")
    m = pd.read_csv(metrics_path)
    labels = [f"run {r.run}" for r in m.itertuples()]
    vals = m["pairwise_acc"].tolist()
    colors = [run_colors.get(r.run, "#7f7f7f") for r in m.itertuples()]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, vals, color=colors)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    plt.axhline(0.5, ls="--", color="gray", label="coin flip (0.50)")
    plt.ylim(0, 1)
    plt.ylabel(f"Pairwise ranking accuracy vs {prop['label']}")
    plt.title("Approach 3 generative: is the LLM ranker a good judge?")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    _save(d, "pairwise3_gen.png")


# --------------------------------------------------------------------------- #
# Cross-approach comparison: fair A2-vs-A3 pairwise, plus 3-way best-found
# --------------------------------------------------------------------------- #

def _mean_std(series):
    a = np.asarray(series, float)
    a = a[~np.isnan(a)]
    return (float(np.mean(a)), float(np.std(a))) if len(a) else (float("nan"), 0.0)


def plots_compare(property_name):
    prop = C.PROPERTIES[property_name]
    start = max(m["score"] for m in get_seeds(property_name))
    b = _baseline_summary(property_name)
    random_baseline = float(b["random_mean"]) if b is not None else None

    m1_path = os.path.join(_results_dir(1, property_name), "metrics1.csv")
    m2_path = os.path.join(_results_dir(2, property_name), "metrics2_gen.csv")
    m3_path = os.path.join(_results_dir(3, property_name), "metrics3_gen.csv")
    if not (os.path.exists(m2_path) and os.path.exists(m3_path)):
        print("Missing metrics2_gen.csv/metrics3_gen.csv. Run _05_analysis.py --approach 2/3 first.")
        return
    m2, m3 = pd.read_csv(m2_path), pd.read_csv(m3_path)
    m1 = pd.read_csv(m1_path) if os.path.exists(m1_path) else None

    acc2_m, acc2_s = _mean_std(m2["pairwise_acc"])
    acc3_m, acc3_s = _mean_std(m3["pairwise_acc"])
    best2_m, best2_s = _mean_std(m2["best_found"])
    best3_m, best3_s = _mean_std(m3["best_found"])

    d = PLOTS_ROOT if property_name == "esol" else os.path.join(PLOTS_ROOT, property_name)
    results_dir = RESULTS_ROOT if property_name == "esol" else os.path.join(RESULTS_ROOT, property_name)
    os.makedirs(results_dir, exist_ok=True)

    summary = pd.DataFrame([
        {"surrogate": "A2 regressor", "n_runs": len(m2), "pairwise_acc_mean": round(acc2_m, 4),
         "pairwise_acc_std": round(acc2_s, 4), "best_found_mean": round(best2_m, 4),
         "best_found_std": round(best2_s, 4)},
        {"surrogate": "A3 ranker", "n_runs": len(m3), "pairwise_acc_mean": round(acc3_m, 4),
         "pairwise_acc_std": round(acc3_s, 4), "best_found_mean": round(best3_m, 4),
         "best_found_std": round(best3_s, 4)},
    ])
    summary.to_csv(os.path.join(results_dir, "comparison_gen.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    labels, colors = ["A2 regressor", "A3 ranker"], ["#1f77b4", "#d62728"]
    ax[0].bar(labels, [acc2_m, acc3_m], yerr=[acc2_s, acc3_s], capsize=6, color=colors)
    ax[0].axhline(0.5, ls="--", color="gray", label="coin flip (0.50)")
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel(f"Pairwise ranking accuracy vs {prop['label']}")
    ax[0].set_title("Fair surrogate quality (both reduced to pairwise)")
    ax[0].legend(fontsize=8)
    ax[0].grid(axis="y", alpha=0.3)

    ax[1].bar(labels, [best2_m, best3_m], yerr=[best2_s, best3_s], capsize=6, color=colors)
    ax[1].axhline(start, ls="--", color="gray", label=f"seed ({start:+.2f})")
    if random_baseline is not None:
        ax[1].axhline(random_baseline, ls=":", color="green", label=f"random ({random_baseline:+.2f})")
    ax[1].set_ylabel(f"Best {prop['label']} found")
    ax[1].set_title("Optimization outcome (best molecule found)")
    ax[1].legend(fontsize=8)
    ax[1].grid(axis="y", alpha=0.3)
    _save(d, "comparison_gen.png")

    if m1 is None:
        print(summary.to_string(index=False))
        return

    # 3-way best-found bar (A1 vs A2 vs A3 vs random baseline).
    labels3 = ["A1 direct\noptimizer", "A2 LLM\nregressor", "A3 LLM\nranker"]
    means3 = [float(m1["best_found"].mean()), best2_m, best3_m]
    spreads3 = [float(m1["best_found"].std(ddof=0)) if len(m1) > 1 else 0.0, best2_s, best3_s]
    colors3 = ["#1f77b4", "#d62728", "#2ca02c"]
    if random_baseline is not None:
        labels3.append("Random\nbaseline")
        means3.append(random_baseline)
        spreads3.append(float(b["random_std"]))
        colors3.append("#7f7f7f")
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels3, means3, yerr=spreads3, capsize=4, color=colors3)
    for bar, v in zip(bars, means3):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:+.2f}", ha="center", fontsize=9)
    plt.ylabel(f"Best {prop['label']} found, mean across runs")
    plt.title("Cross-approach: best molecule found (higher = better)")
    plt.grid(axis="y", alpha=0.3)
    _save(d, "comparison_all.png")
    print(summary.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="Figures for all approaches, any property")
    p.add_argument("--approach", required=True, choices=["1", "2", "3", "compare"])
    p.add_argument("--property", default="esol", choices=list(C.PROPERTIES))
    args = p.parse_args()

    if args.approach == "1":
        plots_approach1(args.property)
    elif args.approach == "2":
        plots_approach2(args.property)
    elif args.approach == "3":
        plots_approach3(args.property)
    elif args.approach == "compare":
        plots_compare(args.property)


if __name__ == "__main__":
    main()
