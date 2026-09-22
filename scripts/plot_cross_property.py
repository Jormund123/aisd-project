"""
One figure comparing ESOL and JNK3 side by side, across all three approaches.

Headline metric (bar height): percent improvement over the seed bar, i.e.
100 * improvement_over_seed / seed_best. This is the one metric that is
directly comparable across two properties with unrelated units (LogS vs a
0-1 probability) -- both seed bars are positive, so "how many times better
than where we started" means the same thing on both sides.

Secondary metrics (text under each bar) are property-appropriate: Approach 1
shows hit-rate (fraction of proposals beating the bar); Approaches 2/3 show
pairwise accuracy against the true oracle. Raw best-found score is annotated
above each bar so the underlying numbers stay visible even though the axis
itself is normalized.

Reads results/approach{1,2,3}[/jnk3]/metrics*.csv and results/baseline_summary[_jnk3].csv
(all produced by _05_analysis.py -- run that first for both properties if these
are missing).

Usage:
    python scripts/plot_cross_property.py
Output:
    plots/cross_property_comparison.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")

APPROACHES = ["1", "2", "3"]
APPROACH_LABELS = {"1": "A1 direct\noptimizer", "2": "A2 LLM\nregressor", "3": "A3 LLM\nranker"}
COLORS = {"1": "#1f77b4", "2": "#d62728", "3": "#2ca02c"}


def _metrics_path(approach, property_name):
    base = os.path.join(RESULTS, f"approach{approach}")
    d = base if property_name == "esol" else os.path.join(base, property_name)
    fname = {"1": "metrics1.csv", "2": "metrics2_gen.csv", "3": "metrics3_gen.csv"}[approach]
    return os.path.join(d, fname)


def _load(property_name):
    seed_best = pd.read_csv(
        os.path.join(RESULTS, f"baseline_summary{'' if property_name == 'esol' else '_' + property_name}.csv")
    )["seed_best"].iloc[0]

    rows = {}
    for a in APPROACHES:
        df = pd.read_csv(_metrics_path(a, property_name))
        best_mean = float(df["best_found"].mean())
        improvement_mean = float(df["improvement_over_seed"].mean())
        pct = 100.0 * improvement_mean / seed_best
        if a == "1":
            secondary = f"hit-rate {df['hit_rate'].mean():.2f}"
        else:
            secondary = f"pairwise acc. {df['pairwise_acc'].mean():.2f}"
        rows[a] = {"best_mean": best_mean, "pct_improvement": pct, "secondary": secondary}
    return rows, seed_best


def main():
    esol, esol_bar = _load("esol")
    jnk3, jnk3_bar = _load("jnk3")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(2)  # ESOL, JNK3
    width = 0.25

    for i, a in enumerate(APPROACHES):
        vals = [esol[a]["pct_improvement"], jnk3[a]["pct_improvement"]]
        secs = [esol[a]["secondary"], jnk3[a]["secondary"]]
        bests = [esol[a]["best_mean"], jnk3[a]["best_mean"]]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=APPROACH_LABELS[a], color=COLORS[a])
        for bar, v, sec, best in zip(bars, vals, secs, bests):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (2 if v >= 0 else -10),
                     f"{v:+.0f}%", ha="center", fontsize=9, fontweight="bold")
            ax.text(bar.get_x() + bar.get_width() / 2, -55,
                     f"best={best:.2f}\n{sec}", ha="center", fontsize=6.5, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels([f"ESOL\n(seed bar {esol_bar:+.3f})", f"JNK3\n(seed bar {jnk3_bar:+.2f})"])
    ax.set_ylabel("Improvement over seed bar (%)")
    ax.set_title("Headline metric: % improvement over the seed bar, ESOL vs. JNK3")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-70, max(esol["1"]["pct_improvement"], 100) * 1.15)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    os.makedirs(PLOTS, exist_ok=True)
    out = os.path.join(PLOTS, "cross_property_comparison.png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
