"""
Cross-approach comparison: Approach 1 vs 2 vs 3 head to head.

Reads the per-run metric tables written by analysis.py / analysis2.py / analysis3.py
(results/metrics1.csv, metrics2.csv, metrics3.csv), aggregates them per approach, and
produces:
  1. results/metrics_all.csv   -- one row per approach (means/maxes across its runs)
                                   plus each approach's headline quality metric.
  2. plots/comparison_all.png  -- best LogS found per approach (mean across runs, with
                                   spread), against the random baseline and seed best.

IMPORTANT caveat (printed too): the three approaches are NOT scored on identical
problems. Approach 1 generates molecules open-ended (no fixed pool, so regret has no
defined ceiling). Approaches 2 and 3 each pick from a 30-molecule pool, but different
pools (A2 flat-random ceiling +0.21, A3 ESOL-stratified ceiling +1.14). So compare
best-LogS and sample efficiency, and read each approach's own accuracy column for
"how good is the LLM at its job"; do not over-read a single ranking of the three.

Run analysis.py, analysis2.py, analysis3.py first. Then:
    python compare_approaches.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

APPROACH_LABEL = {
    1: "A1 direct optimizer",
    2: "A2 LLM regressor (BO)",
    3: "A3 LLM ranker (PBO)",
}
APPROACH_COLOR = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}


def _load(name):
    path = os.path.join(RESULTS_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def _mean(df, col):
    return float(df[col].mean()) if col in df.columns and len(df) else float("nan")


def _quality(approach, df):
    """One-line headline 'is the LLM good at its job?' figure per approach."""
    if approach == 1:
        return f"hit-rate {_mean(df, 'hit_rate'):.2f} (proposals beating seed)"
    if approach == 2:
        return (f"pred MAE {_mean(df, 'pred_mae'):.2f}, R2 {_mean(df, 'pred_r2'):+.2f}, "
                f"sign-acc {_mean(df, 'sign_acc_vs_seed'):.2f}")
    if approach == 3:
        return (f"pairwise-acc {_mean(df, 'pairwise_acc'):.2f}, "
                f"Spearman {_mean(df, 'rank_spearman'):+.2f}, "
                f"NDCG@5 {_mean(df, 'ndcg_at_5'):.2f}")
    return ""


def aggregate():
    rows = []
    # Per-approach metrics now live in results/approachN/; metrics_all.csv and
    # baseline_summary.csv stay at the results/ top level.
    tables = {1: _load(os.path.join("approach1", "metrics1.csv")),
              2: _load(os.path.join("approach2", "metrics2.csv")),
              3: _load(os.path.join("approach3", "metrics3.csv"))}
    for approach, df in tables.items():
        if df is None or len(df) == 0:
            continue
        rows.append({
            "approach": approach,
            "label": APPROACH_LABEL[approach],
            "n_runs": len(df),
            "best_found_mean": round(_mean(df, "best_found"), 4),
            "best_found_max": round(float(df["best_found"].max()), 4),
            "improvement_mean": round(_mean(df, "improvement_over_seed"), 4),
            "success_rate_mean": round(_mean(df, "success_rate"), 4),
            "first_improvement_iter_mean": round(_mean(df, "first_improvement_iter"), 2),
            "invalid_rate_mean": round(_mean(df, "invalid_rate"), 4)
            if "invalid_rate" in df.columns else 0.0,
            "headline_quality": _quality(approach, df),
        })
    return pd.DataFrame(rows), tables


def plot_best(agg, tables):
    labels, means, spreads, colors = [], [], [], []
    for _, r in agg.iterrows():
        a = int(r["approach"])
        df = tables[a]
        labels.append(APPROACH_LABEL[a].replace(" ", "\n", 1))
        means.append(float(r["best_found_mean"]))
        spreads.append(float(df["best_found"].std(ddof=0)) if len(df) > 1 else 0.0)
        colors.append(APPROACH_COLOR[a])

    summary = os.path.join(RESULTS_DIR, "baseline_summary.csv")
    start = None
    if os.path.exists(summary):
        s = pd.read_csv(summary).iloc[0]
        labels.append("Random\nbaseline")
        means.append(float(s["random_mean"]))
        spreads.append(float(s["random_std"]))
        colors.append("#7f7f7f")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, means, yerr=spreads, capsize=4, color=colors)
    for bar, v in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 v + 0.05 + (0.02 if v >= 0 else 0), f"{v:+.2f}",
                 ha="center", fontsize=9)
    plt.ylabel("Best LogS found (ESOL), mean across runs")
    plt.title("Cross-approach: best solubility found (higher = better)")
    plt.grid(axis="y", alpha=0.3)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "comparison_all.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    agg, tables = aggregate()
    if len(agg) == 0:
        print("No metrics CSVs found. Run analysis.py, analysis2.py, analysis3.py "
              "first to generate results/metrics{1,2,3}.csv.")
        return

    path = os.path.join(RESULTS_DIR, "metrics_all.csv")
    agg.to_csv(path, index=False)

    print("Cross-approach comparison (means across each approach's runs):\n")
    cols = ["label", "n_runs", "best_found_mean", "best_found_max",
            "improvement_mean", "success_rate_mean",
            "first_improvement_iter_mean", "invalid_rate_mean"]
    print(agg[cols].to_string(index=False))
    print("\nHeadline LLM quality per approach:")
    for _, r in agg.iterrows():
        print(f"  {r['label']:24s} {r['headline_quality']}")

    print("\nCaveat: approaches are not on identical problems "
          "(A1 open-ended; A2/A3 different pools). Compare best-LogS and sample "
          "efficiency; use each approach's own quality metric for LLM skill.")

    plot_best(agg, tables)
    print(f"Saved: {path}")
    print("Done.")


if __name__ == "__main__":
    main()
