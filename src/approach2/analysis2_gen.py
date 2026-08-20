"""
Analysis for Approach 2 GENERATIVE (latent-space BO, LLM regressor).

Reads results/approach2/approach2gen_{ei,ucb}_run{n}.csv (the real gemini runs) and makes:
  1. convergence2_gen.png   -- best-so-far LogS vs evaluation, all runs overlaid.
  2. topk2_gen.png          -- mean LogS of the top-3 found so far vs round (mentor's
                               optimization curve), EI and UCB overlaid.
  3. calibration2_gen.png   -- LLM predicted LogS vs true ESOL LogS scatter (y=x = perfect).
  4. pairwise2_gen.png      -- surrogate pairwise ranking accuracy vs oracle, per run.
It also writes results/approach2/metrics2_gen.csv and prints a summary.

Usage:  python analysis2_gen.py
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
    SEED_BEST, RANDOM_BASELINE, pairwise_accuracy, top_k_so_far_by_round, regression_error,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "approach2")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "plots", "approach2")

ACQ_COLORS = {"ei": "#1f77b4", "ucb": "#d62728"}
RUN_COLORS = {
    ("ei", 1): "#1f77b4", ("ei", 2): "#4c9bbd", ("ei", 3): "#17becf",
    ("ucb", 1): "#d62728", ("ucb", 2): "#ff7f0e", ("ucb", 3): "#e377c2",
}


def load_runs():
    """Return {(acq, run): DataFrame} for every generative Approach 2 CSV."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "approach2gen_*_run*.csv")))
    runs = {}
    for f in files:
        m = re.search(r"approach2gen_(ei|ucb)_run(\d+)", os.path.basename(f))
        if not m:
            continue
        df = pd.read_csv(f)
        if len(df):
            runs[(m.group(1), int(m.group(2)))] = df
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
    for (acq, run), df in sorted(runs.items()):
        x = [0] + df["eval_idx"].astype(int).tolist()
        y = [SEED_BEST] + df["best_so_far"].astype(float).tolist()
        plt.plot(x, y, marker="o", markersize=3, alpha=0.85,
                 color=RUN_COLORS.get((acq, run), "#7f7f7f"),
                 label=f"{acq.upper()} run {run}")
    plt.axhline(SEED_BEST, ls="--", lw=1, color="gray",
                label=f"seed best ({SEED_BEST:+.2f})")
    plt.axhline(RANDOM_BASELINE, ls=":", lw=1, color="green",
                label=f"random baseline ({RANDOM_BASELINE:+.2f})")
    plt.xlabel("Evaluation (ESOL calls, budget = 15)")
    plt.ylabel("Best-so-far LogS (ESOL)")
    plt.title("Approach 2 generative: best-so-far LogS climbs above seed")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save("convergence2_gen.png")


def plot_topk(runs, k=3):
    plt.figure(figsize=(8, 5))
    for (acq, run), df in sorted(runs.items()):
        r, v = top_k_so_far_by_round(df["round"], df["true_logs_esol"], k=k)
        plt.plot(r, v, marker="s", markersize=5, alpha=0.85,
                 color=RUN_COLORS.get((acq, run), "#7f7f7f"),
                 label=f"{acq.upper()} run {run}")
    plt.axhline(SEED_BEST, ls="--", lw=1, color="gray", label=f"seed ({SEED_BEST:+.2f})")
    plt.xlabel("Round")
    plt.ylabel(f"Mean LogS of top-{k} found so far")
    plt.title(f"Approach 2 generative: top-{k}-so-far vs round")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save("topk2_gen.png")


def _all_points(runs):
    pred, true, acq = [], [], []
    for (a, _), df in runs.items():
        d = df.dropna(subset=["surrogate_label"])
        pred.extend(d["surrogate_label"].astype(float).tolist())
        true.extend(d["true_logs_esol"].astype(float).tolist())
        acq.extend([a] * len(d))
    return np.array(pred), np.array(true), np.array(acq)


def plot_calibration(runs):
    pred, true, acq = _all_points(runs)
    plt.figure(figsize=(6, 6))
    for a in ("ei", "ucb"):
        m = acq == a
        plt.scatter(true[m], pred[m], s=35, alpha=0.7, color=ACQ_COLORS[a],
                    label=f"{a.upper()} picks")
    lo = min(pred.min(), true.min()) - 0.5
    hi = max(pred.max(), true.max()) + 0.5
    plt.plot([lo, hi], [lo, hi], ls="--", color="black", label="perfect (y = x)")
    plt.xlabel("True LogS (ESOL)")
    plt.ylabel("LLM predicted LogS")
    plt.title("Approach 2 generative: LLM prediction vs truth")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    _save("calibration2_gen.png")


def plot_pairwise(per_run):
    plt.figure(figsize=(8, 5))
    labels = [f"{a.upper()}\nrun {r}" for (a, r) in per_run]
    vals = [per_run[(a, r)]["pairwise_acc"] for (a, r) in per_run]
    colors = [ACQ_COLORS[a] for (a, r) in per_run]
    bars = plt.bar(labels, vals, color=colors)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                 ha="center", fontsize=9)
    plt.axhline(0.5, ls="--", color="gray", label="coin flip (0.50)")
    plt.ylim(0, 1)
    plt.ylabel("Pairwise ranking accuracy vs ESOL")
    plt.title("Approach 2 generative: is the LLM regressor a good ranker?")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    _save("pairwise2_gen.png")


def per_run_metrics(runs):
    out = {}
    for (a, r), df in sorted(runs.items()):
        d = df.dropna(subset=["surrogate_label"])
        pred = d["surrogate_label"].astype(float).to_numpy()
        true = d["true_logs_esol"].astype(float).to_numpy()
        err = regression_error(pred, true)
        out[(a, r)] = {
            "acq": a, "run": r,
            "best_found": round(float(df["best_so_far"].astype(float).max()), 4),
            "improvement_over_seed":
                round(float(df["best_so_far"].astype(float).max()) - SEED_BEST, 4),
            "pairwise_acc": round(pairwise_accuracy(pred, true), 4),
            "pred_mae": round(err["mae"], 4),
            "pred_rmse": round(err["rmse"], 4),
            "pred_bias": round(err["bias"], 4),
            "n_evals": len(df),
        }
    return out


def main():
    runs = load_runs()
    if not runs:
        print("No generative Approach 2 CSVs found (results/approach2/approach2gen_*).")
        return
    print(f"Loaded {len(runs)} run(s): "
          + ", ".join(f"{a}{r}" for a, r in sorted(runs)))

    plot_convergence(runs)
    plot_topk(runs)
    plot_calibration(runs)
    per_run = per_run_metrics(runs)
    plot_pairwise(per_run)

    df = pd.DataFrame(list(per_run.values()))
    path = os.path.join(RESULTS_DIR, "metrics2_gen.csv")
    df.to_csv(path, index=False)

    pred, true, _ = _all_points(runs)
    print(f"\nSeed best = {SEED_BEST:+.3f} | random baseline = {RANDOM_BASELINE:+.3f}")
    print(df.to_string(index=False))
    print(f"\nPooled surrogate pairwise accuracy = {pairwise_accuracy(pred, true):.4f}")
    print(f"Pooled MAE = {regression_error(pred, true)['mae']:.4f}")
    print(f"Saved metrics: {path}\nDone.")


if __name__ == "__main__":
    main()
