"""
Post-experiment analysis for Approach 3 (preferential BO with the LLM as a ranker).

Reads results/approach3_run{n}.csv (the optimization log) and
results/approach3_run{n}_pool.csv (the final full-pool BT ranking) and produces:
  1. convergence3.png    -- best-so-far LogS vs iteration, runs overlaid, with seed
                            best floor and the pool's true ESOL ceiling.
  2. ranking_quality3.png -- final BT utility vs true ESOL LogS scatter (one series per
                            run). Annotated with Spearman / Kendall / pairwise accuracy:
                            the headline "is the LLM a good pairwise ranker?" answer.
  3. comparison3.png      -- best LogS per run vs random baseline vs pool ceiling vs
                            seed best.

By default only genuine manual runs are analyzed; oracle self-test runs (llm==oracle)
are skipped unless --include-auto is passed.

Usage:
    python analysis3.py
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from esol import calculate_esol
from seed_molecules import SEED_MOLECULES
from pairwise import ranking_metrics
import metrics as M

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "approach3")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "plots", "approach3")
# baseline_summary.csv is shared across approaches; it stays at the results/ top level.
SHARED_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

RUN_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def seed_best():
    return max(calculate_esol(m["smiles"])["logs_esol"] for m in SEED_MOLECULES)


def load_runs(include_auto=False):
    """Return {run: (log_df, pool_df)} for every Approach 3 run found."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "approach3_run*.csv"))):
        if f.endswith("_pool.csv"):
            continue
        m = re.search(r"approach3_run(\d+)\.csv$", os.path.basename(f))
        if not m:
            continue
        log = pd.read_csv(f)
        if len(log) == 0:
            continue
        if not include_auto and str(log["llm"].iloc[0]).lower() == "oracle":
            continue
        pool_f = f.replace(".csv", "_pool.csv")
        pool = pd.read_csv(pool_f) if os.path.exists(pool_f) else None
        runs[int(m.group(1))] = (log, pool)
    return runs


def _color(i):
    return RUN_COLORS[i % len(RUN_COLORS)]


def pool_ceiling(runs):
    """Best true ESOL LogS in the pool (read from any run's pool file)."""
    for _, pool in runs.values():
        if pool is not None:
            return float(pool["true_logs_esol"].max())
    return None


def plot_convergence(runs, start):
    ceiling = pool_ceiling(runs)
    plt.figure(figsize=(8, 5))
    for i, (run, (log, _)) in enumerate(sorted(runs.items())):
        iters = [0] + log["iteration"].tolist()
        best = [start] + log["best_so_far"].astype(float).tolist()
        plt.plot(iters, best, marker="o", markersize=4, color=_color(i),
                 label=f"run {run}")
    plt.axhline(start, ls="--", lw=1, color="gray", label=f"seed best ({start:+.2f})")
    if ceiling is not None:
        plt.axhline(ceiling, ls=":", lw=1.5, color="green",
                    label=f"pool ceiling ({ceiling:+.2f})")
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far LogS (ESOL)")
    plt.title("Approach 3: convergence of best-so-far LogS")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save("convergence3.png")


def plot_ranking_quality(runs):
    plt.figure(figsize=(7, 6))
    metric_lines = []
    for i, (run, (_, pool)) in enumerate(sorted(runs.items())):
        if pool is None:
            continue
        true = pool["true_logs_esol"].astype(float).to_numpy()
        util = pool["final_utility"].astype(float).to_numpy()
        plt.scatter(true, util, s=35, alpha=0.7, color=_color(i), label=f"run {run}")
        m = ranking_metrics(util, true)
        metric_lines.append(f"run {run}: rho={m['spearman']:+.2f} "
                            f"tau={m['kendall']:+.2f} pacc={m['pairwise_accuracy']:.2f}")
    plt.xlabel("True LogS (ESOL)")
    plt.ylabel("LLM pairwise-ranking utility (Bradley-Terry)")
    plt.title("Approach 3: is the LLM a good pairwise ranker?")
    if metric_lines:
        plt.gcf().text(0.13, 0.86 - 0.04 * len(metric_lines),
                       "\n".join(metric_lines), fontsize=8,
                       bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=0.3)
    _save("ranking_quality3.png")


def plot_comparison(runs, start):
    ceiling = pool_ceiling(runs)
    labels, values, colors = [], [], []
    for i, (run, (log, _)) in enumerate(sorted(runs.items())):
        labels.append(f"A3\nrun {run}")
        values.append(float(log["best_so_far"].astype(float).max()))
        colors.append(_color(i))

    summary = os.path.join(SHARED_RESULTS_DIR, "baseline_summary.csv")
    if os.path.exists(summary):
        s = pd.read_csv(summary).iloc[0]
        labels.append("Random\nbaseline")
        values.append(float(s["random_mean"]))
        colors.append("#7f7f7f")

    if ceiling is not None:
        labels.append("Pool\nceiling")
        values.append(ceiling)
        colors.append("#2ca02c")
    labels.append("Seed best")
    values.append(start)
    colors.append("#bbbbbb")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:+.2f}",
                 ha="center", fontsize=9)
    plt.ylabel("Best LogS found (ESOL)")
    plt.title("Approach 3: runs vs baseline vs ceiling")
    plt.grid(axis="y", alpha=0.3)
    _save("comparison3.png")


def _save(name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-auto", action="store_true",
                    help="also analyze oracle self-test runs (llm==oracle)")
    args = ap.parse_args()

    runs = load_runs(include_auto=args.include_auto)
    if not runs:
        print("No Approach 3 manual runs found in results/ "
              "(use --include-auto to analyze oracle self-tests).")
        return
    print(f"Loaded {len(runs)} run(s): {sorted(runs)}")

    start = seed_best()
    ceiling = pool_ceiling(runs)
    print(f"seed best    = {start:+.4f}")
    if ceiling is not None:
        print(f"pool ceiling = {ceiling:+.4f}")

    plot_convergence(runs, start)
    plot_ranking_quality(runs)
    plot_comparison(runs, start)

    report_metrics(runs, start, ceiling)
    print("Done.")


def run_metrics(run, log, pool, start, ceiling):
    """Per-run optimization + ranking-accuracy metrics for an Approach 3 run."""
    best_sf = log["best_so_far"].astype(float).to_numpy()
    iters = log["iteration"].astype(int).tolist()
    bf = M.best_found(best_sf, start)
    row = {
        "approach": 3, "run": run,
        "best_found": round(bf, 4),
        "improvement_over_seed": round(M.improvement_over_seed(bf, start), 4),
        "simple_regret": round(M.simple_regret(bf, ceiling), 4),
        "normalized_score": round(M.normalized_score(bf, start, ceiling), 4),
        "first_improvement_iter": M.first_improvement_iter(best_sf, start, iters),
        "iters_to_ceiling": M.iters_to_ceiling(best_sf, ceiling, iters),
        "success_rate": round(M.success_rate(best_sf, start), 4),
        "n_evals": len(log),
    }
    if pool is not None:
        util = pool["final_utility"].astype(float).to_numpy()
        true = pool["true_logs_esol"].astype(float).to_numpy()
        rm = ranking_metrics(util, true)
        row.update({
            "rank_spearman": round(rm["spearman"], 4),
            "rank_kendall": round(rm["kendall"], 4),
            "pairwise_acc": round(rm["pairwise_accuracy"], 4),
            "top1_hit": M.top1_hit(util, true),
            "ndcg_at_5": round(M.ndcg_at_k(util, true, k=5), 4),
        })
    return row


def report_metrics(runs, start, ceiling):
    rows = [run_metrics(run, log, pool, start, ceiling)
            for run, (log, pool) in sorted(runs.items())]
    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "metrics3.csv")
    out.to_csv(path, index=False)
    cstr = f"{ceiling:+.4f}" if ceiling is not None else "n/a"
    print(f"\nApproach 3 per-run metrics (seed best = {start:+.4f}, "
          f"pool ceiling = {cstr}):")
    cols = ["run", "best_found", "simple_regret", "normalized_score", "success_rate",
            "rank_spearman", "rank_kendall", "pairwise_acc", "top1_hit", "ndcg_at_5"]
    cols = [c for c in cols if c in out.columns]
    print(out[cols].to_string(index=False))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
