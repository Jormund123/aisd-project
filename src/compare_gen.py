"""
Head-to-head comparison of the two generative surrogates (the mentor's fair test).

A regressor (Approach 2) and a ranker (Approach 3) do different jobs, so they cannot be
compared by their raw outputs. The fix he asked for: reduce BOTH to pairwise orderings and
score each against the ESOL oracle's ordering. Then a regressor's numbers and a ranker's
duels sit on ONE axis (pairwise ranking accuracy), so we can finally say which understands
solubility better. We also compare best-LogS-found (the optimization outcome).

Reads the metrics written by analysis2_gen.py and analysis3_gen.py, so run those first.
Writes plots/comparison_gen.png and results/comparison_gen.csv.

Usage:  python compare_gen.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from analysis_common import SEED_BEST, RANDOM_BASELINE  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS = os.path.join(os.path.dirname(__file__), "..", "plots")
M2 = os.path.join(RESULTS, "approach2", "metrics2_gen.csv")
M3 = os.path.join(RESULTS, "approach3", "metrics3_gen.csv")


def _mean_std(series):
    a = np.asarray(series, float)
    a = a[~np.isnan(a)]
    return (float(np.mean(a)), float(np.std(a))) if len(a) else (float("nan"), 0.0)


def main():
    if not (os.path.exists(M2) and os.path.exists(M3)):
        print("Missing metrics. Run analysis2_gen.py and analysis3_gen.py first.")
        return
    m2 = pd.read_csv(M2)
    m3 = pd.read_csv(M3)

    acc2_m, acc2_s = _mean_std(m2["pairwise_acc"])
    acc3_m, acc3_s = _mean_std(m3["pairwise_acc"])
    best2_m, best2_s = _mean_std(m2["best_found"])
    best3_m, best3_s = _mean_std(m3["best_found"])

    summary = pd.DataFrame([
        {"surrogate": "A2 regressor (LLM predicts LogS)",
         "n_runs": len(m2), "pairwise_acc_mean": round(acc2_m, 4),
         "pairwise_acc_std": round(acc2_s, 4),
         "best_found_mean": round(best2_m, 4), "best_found_std": round(best2_s, 4)},
        {"surrogate": "A3 ranker (LLM judges pairs)",
         "n_runs": len(m3), "pairwise_acc_mean": round(acc3_m, 4),
         "pairwise_acc_std": round(acc3_s, 4),
         "best_found_mean": round(best3_m, 4), "best_found_std": round(best3_s, 4)},
    ])
    out_csv = os.path.join(RESULTS, "comparison_gen.csv")
    summary.to_csv(out_csv, index=False)

    # Two panels: fair surrogate quality (pairwise acc) + optimization outcome (best LogS).
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["A2 regressor", "A3 ranker"]
    colors = ["#1f77b4", "#d62728"]

    ax[0].bar(labels, [acc2_m, acc3_m], yerr=[acc2_s, acc3_s], capsize=6, color=colors)
    ax[0].axhline(0.5, ls="--", color="gray", label="coin flip (0.50)")
    for i, (m, s) in enumerate([(acc2_m, acc2_s), (acc3_m, acc3_s)]):
        ax[0].text(i, m + s + 0.02, f"{m:.2f}", ha="center", fontsize=10)
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("Pairwise ranking accuracy vs ESOL")
    ax[0].set_title("Fair surrogate quality (both reduced to pairwise)")
    ax[0].legend(fontsize=8)
    ax[0].grid(axis="y", alpha=0.3)

    ax[1].bar(labels, [best2_m, best3_m], yerr=[best2_s, best3_s], capsize=6, color=colors)
    ax[1].axhline(SEED_BEST, ls="--", color="gray", label=f"seed ({SEED_BEST:+.2f})")
    ax[1].axhline(RANDOM_BASELINE, ls=":", color="green",
                  label=f"random ({RANDOM_BASELINE:+.2f})")
    for i, (m, s) in enumerate([(best2_m, best2_s), (best3_m, best3_s)]):
        ax[1].text(i, m + s + 0.03, f"{m:+.2f}", ha="center", fontsize=10)
    ax[1].set_ylabel("Best LogS found (ESOL)")
    ax[1].set_title("Optimization outcome (best molecule found)")
    ax[1].legend(fontsize=8)
    ax[1].grid(axis="y", alpha=0.3)

    os.makedirs(PLOTS, exist_ok=True)
    path = os.path.join(PLOTS, "comparison_gen.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    print(summary.to_string(index=False))
    print(f"\nSaved: {path}\nSaved: {out_csv}\nDone.")


if __name__ == "__main__":
    main()
