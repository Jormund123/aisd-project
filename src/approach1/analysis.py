"""
Post-experiment analysis for Approach 1.

Reads results/approach1_variant*_run*.csv and produces four plots in plots/:
  1. convergence.png  -- best-so-far LogS vs iteration, variants overlaid.
  2. invalid_rate.png -- fraction of proposals that failed validation, per variant.
  3. diversity.png    -- mean pairwise Tanimoto similarity of valid proposals
                         per variant (lower = more diverse).
  4. comparison.png   -- best LogS per variant vs random baseline vs AqSolDB ceiling.

Usage:
    python analysis.py
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from esol import calculate_esol
from seed_molecules import SEED_MOLECULES
import metrics as M

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "approach1")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "plots", "approach1")
# baseline_summary.csv is shared across approaches; it stays at the results/ top level.
SHARED_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

VARIANT_COLORS = {"A": "#d62728", "B": "#1f77b4", "C": "#2ca02c"}

# Convergence is per-run: every run gets its own distinct color. Known runs are
# fixed here so figures are reproducible; any extra run falls back to the tab10
# cycle by sorted order.
RUN_COLORS = {
    ("A", 1): "#d62728",  # red
    ("A", 2): "#ff7f0e",  # orange
    ("B", 1): "#1f77b4",  # blue
    ("B", 2): "#9467bd",  # purple
    ("C", 1): "#2ca02c",  # green
    ("C", 2): "#8c564b",  # brown
}
_FALLBACK_CYCLE = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def run_color(variant, run, ordinal):
    """Distinct color per (variant, run). Falls back to a cycle for unknown runs."""
    if (variant, run) in RUN_COLORS:
        return RUN_COLORS[(variant, run)]
    return _FALLBACK_CYCLE[ordinal % len(_FALLBACK_CYCLE)]


def seed_best():
    """Best ESOL LogS among the seed molecules (the optimizer's iteration-0 start)."""
    vals = [calculate_esol(m["smiles"])["logs_esol"] for m in SEED_MOLECULES]
    return max(vals)


def load_runs():
    """Return {(variant, run): DataFrame} for every result CSV found."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "approach1_variant*_run*.csv")))
    runs = {}
    for f in files:
        m = re.search(r"variant([A-C])_run(\d+)", os.path.basename(f))
        if not m:
            continue
        df = pd.read_csv(f)
        if len(df) == 0:
            continue
        runs[(m.group(1), int(m.group(2)))] = df
    return runs


def is_valid(series):
    """CSV 'valid' column may be bool or 'True'/'False' string. Normalize to bool."""
    return series.astype(str).str.strip().str.lower() == "true"


def plot_convergence(runs, start):
    plt.figure(figsize=(8, 5))
    for ordinal, ((variant, run), df) in enumerate(sorted(runs.items())):
        iters = [0] + df["iteration"].tolist()
        best = [start] + df["best_so_far"].astype(float).tolist()
        plt.plot(iters, best, marker="o", markersize=3,
                 color=run_color(variant, run, ordinal),
                 label=f"Variant {variant} run {run}")
    plt.axhline(start, ls="--", lw=1, color="gray", label=f"seed best ({start:+.2f})")
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far LogS (ESOL)")
    plt.title("Approach 1: convergence of best-so-far LogS")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save("convergence.png")


def plot_invalid_rate(runs):
    by_variant = {}
    for (variant, _), df in runs.items():
        valid = is_valid(df["valid"])
        by_variant.setdefault(variant, []).append(1 - valid.mean())
    variants = sorted(by_variant)
    rates = [np.mean(by_variant[v]) for v in variants]
    plt.figure(figsize=(6, 4))
    plt.bar(variants, rates, color=[VARIANT_COLORS[v] for v in variants])
    for i, r in enumerate(rates):
        plt.text(i, r + 0.005, f"{r:.0%}", ha="center", fontsize=9)
    plt.ylabel("Invalid / failed proposal rate")
    plt.title("Invalid-SMILES (failed) rate per variant")
    plt.ylim(0, max(rates + [0.1]) * 1.3)
    _save("invalid_rate.png")


def _fingerprints(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    return fps


def mean_pairwise_similarity(smiles_list):
    fps = _fingerprints(smiles_list)
    if len(fps) < 2:
        return np.nan
    sims = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    return float(np.mean(sims)) if sims else np.nan


def plot_diversity(runs):
    by_variant = {}
    for (variant, _), df in runs.items():
        valid = is_valid(df["valid"])
        smiles = df.loc[valid, "canonical_smiles"].dropna().tolist()
        sim = mean_pairwise_similarity(smiles)
        if not np.isnan(sim):
            by_variant.setdefault(variant, []).append(sim)
    variants = sorted(by_variant)
    sims = [np.mean(by_variant[v]) for v in variants]
    plt.figure(figsize=(6, 4))
    plt.bar(variants, sims, color=[VARIANT_COLORS[v] for v in variants])
    for i, s in enumerate(sims):
        plt.text(i, s + 0.01, f"{s:.2f}", ha="center", fontsize=9)
    plt.ylabel("Mean pairwise Tanimoto similarity")
    plt.title("Proposal similarity per variant (lower = more diverse)")
    plt.ylim(0, 1)
    _save("diversity.png")


def plot_comparison(runs, start):
    labels, values, colors = [], [], []
    for variant in sorted({v for v, _ in runs}):
        best = max(
            float(df["best_so_far"].astype(float).max())
            for (v, _), df in runs.items() if v == variant
        )
        labels.append(f"Variant {variant}")
        values.append(best)
        colors.append(VARIANT_COLORS[variant])

    summary_path = os.path.join(SHARED_RESULTS_DIR, "baseline_summary.csv")
    if os.path.exists(summary_path):
        s = pd.read_csv(summary_path).iloc[0]
        labels += ["Random\nbaseline", "AqSolDB\nESOL ceiling"]
        values += [float(s["random_mean"]), float(s["esol_ceiling"])]
        colors += ["#7f7f7f", "#000000"]

    labels.append("Seed best")
    values.append(start)
    colors.append("#bbbbbb")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                 f"{v:+.2f}", ha="center", fontsize=8)
    plt.ylabel("Best LogS found (ESOL)")
    plt.title("Approach 1: optimizer vs baseline vs ceiling")
    plt.grid(axis="y", alpha=0.3)
    _save("comparison.png")


def _save(name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def run_metrics(variant, run, df, start):
    """Per-run metrics for an Approach 1 run.

    Approach 1 is open-ended generation (no fixed candidate pool), so regret and
    normalized score are undefined (ceiling=None). Its quality signal is hit-rate:
    how often a proposal's true ESOL actually beat the seed best.
    """
    best_sf = df["best_so_far"].astype(float).to_numpy()
    iters = df["iteration"].astype(int).tolist()
    valid = is_valid(df["valid"])
    true_vals = df.loc[valid, "logs_esol"].astype(float).to_numpy()
    smiles = df.loc[valid, "canonical_smiles"].dropna().tolist()
    bf = M.best_found(best_sf, start)
    return {
        "approach": 1, "variant": variant, "run": run,
        "best_found": round(bf, 4),
        "improvement_over_seed": round(M.improvement_over_seed(bf, start), 4),
        "simple_regret": float("nan"),          # no fixed ceiling for open generation
        "normalized_score": float("nan"),
        "first_improvement_iter": M.first_improvement_iter(best_sf, start, iters),
        "iters_to_ceiling": None,
        "success_rate": round(M.success_rate(best_sf, start), 4),
        "hit_rate": round(M.hit_rate(true_vals, start), 4),
        "invalid_rate": round(1 - valid.mean(), 4),
        "diversity_tanimoto": round(mean_pairwise_similarity(smiles), 4),
        "n_evals": len(df),
    }


def report_metrics(runs, start):
    rows = [run_metrics(v, r, df, start) for (v, r), df in sorted(runs.items())]
    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "metrics1.csv")
    out.to_csv(path, index=False)
    print(f"\nApproach 1 per-run metrics (seed best = {start:+.4f}):")
    cols = ["variant", "run", "best_found", "improvement_over_seed", "success_rate",
            "hit_rate", "invalid_rate", "diversity_tanimoto", "first_improvement_iter"]
    print(out[cols].to_string(index=False))
    print(f"Saved: {path}")


def main():
    runs = load_runs()
    if not runs:
        print("No result CSVs found in results/. Run the optimizer first.")
        return
    print(f"Loaded {len(runs)} run(s): "
          + ", ".join(f"{v}{r}" for v, r in sorted(runs)))
    start = seed_best()
    plot_convergence(runs, start)
    plot_invalid_rate(runs)
    plot_diversity(runs)
    plot_comparison(runs, start)
    report_metrics(runs, start)
    print("Done.")


if __name__ == "__main__":
    main()
