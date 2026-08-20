"""
Analysis for Approach 3 GENERATIVE (latent-space preferential BO, LLM pairwise ranker).

Reads results/approach3/approach3gen_run{n}.csv (optimization log) and
results/approach3/approach3gen_run{n}_ranking.csv (final Bradley-Terry utility vs ESOL)
and makes:
  1. convergence3_gen.png     -- best-so-far LogS vs evaluation, runs overlaid.
  2. topk3_gen.png            -- mean LogS of the top-3 found so far vs round.
  3. ranking_quality3_gen.png -- Bradley-Terry utility vs true ESOL LogS scatter per run.
  4. pairwise3_gen.png        -- ranker pairwise accuracy vs oracle, per run.
It also writes results/approach3/metrics3_gen.csv and prints a summary.

Usage:  python analysis3_gen.py
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis_common import (  # noqa: E402
    SEED_BEST, RANDOM_BASELINE, pairwise_accuracy, top_k_so_far_by_round,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "approach3")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "plots", "approach3")

RUN_COLORS = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}


def load_runs():
    """Return {run: (opt_df, ranking_df)} for every generative Approach 3 run."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "approach3gen_run*.csv")))
    runs = {}
    for f in files:
        base = os.path.basename(f)
        if "ranking" in base:
            continue
        m = re.search(r"approach3gen_run(\d+)\.csv", base)
        if not m:
            continue
        run = int(m.group(1))
        opt = pd.read_csv(f)
        rank_path = os.path.join(RESULTS_DIR, f"approach3gen_run{run}_ranking.csv")
        rank = pd.read_csv(rank_path) if os.path.exists(rank_path) else None
        if len(opt):
            runs[run] = (opt, rank)
    return runs


def _save(name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_convergence(runs):
    plt.figure(figsize=(8, 5))
    for run, (opt, _) in sorted(runs.items()):
        x = [0] + opt["eval_idx"].astype(int).tolist()
        y = [SEED_BEST] + opt["best_so_far"].astype(float).tolist()
        plt.plot(x, y, marker="o", markersize=3, color=RUN_COLORS.get(run, "#7f7f7f"),
                 label=f"run {run}")
    plt.axhline(SEED_BEST, ls="--", lw=1, color="gray",
                label=f"seed best ({SEED_BEST:+.2f})")
    plt.axhline(RANDOM_BASELINE, ls=":", lw=1, color="green",
                label=f"random baseline ({RANDOM_BASELINE:+.2f})")
    plt.xlabel("Evaluation (ESOL calls, budget = 15)")
    plt.ylabel("Best-so-far LogS (ESOL)")
    plt.title("Approach 3 generative: best-so-far LogS climbs above seed")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save("convergence3_gen.png")


def plot_topk(runs, k=3):
    plt.figure(figsize=(8, 5))
    for run, (opt, _) in sorted(runs.items()):
        r, v = top_k_so_far_by_round(opt["round"], opt["true_logs_esol"], k=k)
        plt.plot(r, v, marker="s", markersize=5, color=RUN_COLORS.get(run, "#7f7f7f"),
                 label=f"run {run}")
    plt.axhline(SEED_BEST, ls="--", lw=1, color="gray", label=f"seed ({SEED_BEST:+.2f})")
    plt.xlabel("Round")
    plt.ylabel(f"Mean LogS of top-{k} found so far")
    plt.title(f"Approach 3 generative: top-{k}-so-far vs round")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save("topk3_gen.png")


def plot_ranking_quality(runs):
    plt.figure(figsize=(6, 6))
    for run, (_, rank) in sorted(runs.items()):
        if rank is None:
            continue
        plt.scatter(rank["true_logs_esol"], rank["bt_utility"], s=35, alpha=0.7,
                    color=RUN_COLORS.get(run, "#7f7f7f"), label=f"run {run}")
    plt.xlabel("True LogS (ESOL)")
    plt.ylabel("Bradley-Terry utility (ranker strength)")
    plt.title("Approach 3 generative: ranker utility vs truth")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save("ranking_quality3_gen.png")


def plot_pairwise(per_run):
    plt.figure(figsize=(7, 5))
    labels = [f"run {r}" for r in per_run]
    vals = [per_run[r]["pairwise_acc"] for r in per_run]
    colors = [RUN_COLORS.get(r, "#7f7f7f") for r in per_run]
    bars = plt.bar(labels, vals, color=colors)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                 ha="center", fontsize=9)
    plt.axhline(0.5, ls="--", color="gray", label="coin flip (0.50)")
    plt.ylim(0, 1)
    plt.ylabel("Pairwise ranking accuracy vs ESOL")
    plt.title("Approach 3 generative: is the LLM ranker a good judge?")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    _save("pairwise3_gen.png")


def per_run_metrics(runs):
    out = {}
    for run, (opt, rank) in sorted(runs.items()):
        pacc = float("nan")
        if rank is not None:
            pacc = pairwise_accuracy(rank["bt_utility"], rank["true_logs_esol"])
        out[run] = {
            "run": run,
            "best_found": round(float(opt["best_so_far"].astype(float).max()), 4),
            "improvement_over_seed":
                round(float(opt["best_so_far"].astype(float).max()) - SEED_BEST, 4),
            "pairwise_acc": round(pacc, 4),
            "n_duels": int(opt["n_duels_cum"].astype(int).max()),
            "n_evals": len(opt),
        }
    return out


def main():
    runs = load_runs()
    if not runs:
        print("No generative Approach 3 CSVs found (results/approach3/approach3gen_*).")
        return
    print(f"Loaded {len(runs)} run(s): " + ", ".join(f"run{r}" for r in sorted(runs)))

    plot_convergence(runs)
    plot_topk(runs)
    plot_ranking_quality(runs)
    per_run = per_run_metrics(runs)
    plot_pairwise(per_run)

    df = pd.DataFrame(list(per_run.values()))
    path = os.path.join(RESULTS_DIR, "metrics3_gen.csv")
    df.to_csv(path, index=False)
    print(f"\nSeed best = {SEED_BEST:+.3f} | random baseline = {RANDOM_BASELINE:+.3f}")
    print(df.to_string(index=False))
    print(f"Saved metrics: {path}\nDone.")


if __name__ == "__main__":
    main()
