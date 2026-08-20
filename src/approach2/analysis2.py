"""
Post-experiment analysis for Approach 2 (BO with LLM as regressor).

Reads results/approach2_{acq}_run{n}.csv and produces four plots in plots/:
  1. convergence2.png   -- best-so-far LogS vs iteration, EI vs UCB runs overlaid,
                           with seed-best floor and the pool's true ESOL ceiling.
  2. calibration2.png   -- LLM predicted LogS (mu) vs true ESOL LogS scatter. If the
                           LLM were a good regressor, points sit on the y=x line.
  3. confidence_error2.png -- absolute prediction error vs LLM confidence. If
                           confidence were calibrated, high confidence => low error.
  4. comparison2.png    -- best LogS per acquisition vs random baseline vs pool
                           ceiling vs seed best.

It also prints summary numbers (pool ceiling, MAE, correlation) used in the writeup.

Usage:
    python analysis2.py
"""

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
from optimizer_approach2 import build_pool, build_seed_observed
import metrics as M

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "approach2")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "plots", "approach2")
# baseline_summary.csv is shared across approaches; it stays at the results/ top level.
SHARED_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

# One color per (acq, run). EI shades of blue, UCB shades of orange/red.
RUN_COLORS = {
    ("ei", 1): "#1f77b4",   # blue
    ("ei", 2): "#17becf",   # cyan
    ("ucb", 1): "#d62728",  # red
    ("ucb", 2): "#ff7f0e",  # orange
}
ACQ_COLORS = {"ei": "#1f77b4", "ucb": "#d62728"}


def seed_best():
    """Best ESOL LogS among the seed molecules (the optimizer's iteration-0 start)."""
    return max(calculate_esol(m["smiles"])["logs_esol"] for m in SEED_MOLECULES)


def pool_ceiling(pool_size=30, pool_seed=0):
    """True ESOL of the best molecule in the candidate pool (the real ceiling any
    acquisition function could reach on this pool)."""
    observed = build_seed_observed()
    pool = build_pool(pool_size, pool_seed, {m["smiles"] for m in observed})
    vals = [(calculate_esol(c["smiles"])["logs_esol"], c["name"]) for c in pool]
    vals.sort(reverse=True)
    return vals[0][0], vals[0][1], vals


def load_runs():
    """Return {(acq, run): DataFrame} for every Approach 2 result CSV found."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "approach2_*_run*.csv")))
    runs = {}
    for f in files:
        m = re.search(r"approach2_(ei|ucb)_run(\d+)", os.path.basename(f))
        if not m:
            continue
        df = pd.read_csv(f)
        if len(df) == 0:
            continue
        runs[(m.group(1), int(m.group(2)))] = df
    return runs


def plot_convergence(runs, start, ceiling):
    plt.figure(figsize=(8, 5))
    for (acq, run), df in sorted(runs.items()):
        iters = [0] + df["iteration"].tolist()
        best = [start] + df["best_so_far"].astype(float).tolist()
        plt.plot(iters, best, marker="o", markersize=4,
                 color=RUN_COLORS.get((acq, run), "#7f7f7f"),
                 label=f"{acq.upper()} run {run}")
    plt.axhline(start, ls="--", lw=1, color="gray", label=f"seed best ({start:+.2f})")
    plt.axhline(ceiling, ls=":", lw=1.5, color="green",
                label=f"pool ceiling ({ceiling:+.2f})")
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far LogS (ESOL)")
    plt.title("Approach 2: convergence of best-so-far LogS (EI vs UCB)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save("convergence2.png")


def _all_points(runs):
    """Flatten every (mu, true, confidence, acq) tested across all runs."""
    mu, true, conf, acq = [], [], [], []
    for (a, _), df in runs.items():
        mu.extend(df["predicted_logs"].astype(float).tolist())
        true.extend(df["true_logs_esol"].astype(float).tolist())
        conf.extend(df["confidence"].astype(float).tolist())
        acq.extend([a] * len(df))
    return (np.array(mu), np.array(true), np.array(conf), np.array(acq))


def plot_calibration(runs):
    mu, true, conf, acq = _all_points(runs)
    plt.figure(figsize=(6, 6))
    for a in ("ei", "ucb"):
        m = acq == a
        plt.scatter(true[m], mu[m], s=40, alpha=0.7, color=ACQ_COLORS[a],
                    label=f"{a.upper()} picks")
    lo = min(mu.min(), true.min()) - 0.5
    hi = max(mu.max(), true.max()) + 0.5
    plt.plot([lo, hi], [lo, hi], ls="--", color="black", label="perfect (y = x)")
    plt.xlabel("True LogS (ESOL)")
    plt.ylabel("LLM predicted LogS (mu)")
    plt.title("Approach 2: LLM prediction vs truth")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    _save("calibration2.png")
    mae = float(np.mean(np.abs(mu - true)))
    corr = float(np.corrcoef(mu, true)[0, 1]) if len(mu) > 1 else float("nan")
    bias = float(np.mean(mu - true))
    return mae, corr, bias


def plot_confidence_error(runs):
    mu, true, conf, acq = _all_points(runs)
    err = np.abs(mu - true)
    plt.figure(figsize=(7, 5))
    for a in ("ei", "ucb"):
        m = acq == a
        plt.scatter(conf[m], err[m], s=40, alpha=0.7, color=ACQ_COLORS[a],
                    label=f"{a.upper()} picks")
    # trend line: mean error per confidence value
    xs = sorted(set(conf.tolist()))
    ys = [float(np.mean(err[conf == x])) for x in xs]
    plt.plot(xs, ys, color="black", lw=1.5, marker="s", label="mean error")
    plt.xlabel("LLM confidence (1-10, higher = more sure)")
    plt.ylabel("Absolute prediction error |mu - true|")
    plt.title("Approach 2: is confidence calibrated?")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save("confidence_error2.png")
    # correlation between confidence and error (want negative if calibrated)
    return float(np.corrcoef(conf, err)[0, 1]) if len(conf) > 1 else float("nan")


def plot_comparison(runs, start, ceiling):
    labels, values, colors = [], [], []
    for acq in ("ei", "ucb"):
        bests = [float(df["best_so_far"].astype(float).max())
                 for (a, _), df in runs.items() if a == acq]
        if bests:
            labels.append(acq.upper())
            values.append(max(bests))
            colors.append(ACQ_COLORS[acq])

    summary_path = os.path.join(SHARED_RESULTS_DIR, "baseline_summary.csv")
    if os.path.exists(summary_path):
        s = pd.read_csv(summary_path).iloc[0]
        labels.append("Random\nbaseline")
        values.append(float(s["random_mean"]))
        colors.append("#7f7f7f")

    labels += ["Pool\nceiling", "Seed best"]
    values += [ceiling, start]
    colors += ["#2ca02c", "#bbbbbb"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                 f"{v:+.2f}", ha="center", fontsize=9)
    plt.ylabel("Best LogS found (ESOL)")
    plt.title("Approach 2: EI vs UCB vs baseline vs ceiling")
    plt.grid(axis="y", alpha=0.3)
    _save("comparison2.png")


def _save(name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def run_metrics(acq, run, df, start, ceiling):
    """Per-run optimization + regression-accuracy metrics for an Approach 2 run."""
    best_sf = df["best_so_far"].astype(float).to_numpy()
    iters = df["iteration"].astype(int).tolist()
    pred = df["predicted_logs"].astype(float).to_numpy()
    true = df["true_logs_esol"].astype(float).to_numpy()
    bf = M.best_found(best_sf, start)
    reg = M.regression_metrics(pred, true)
    return {
        "approach": 2, "acq": acq, "run": run,
        "best_found": round(bf, 4),
        "improvement_over_seed": round(M.improvement_over_seed(bf, start), 4),
        "simple_regret": round(M.simple_regret(bf, ceiling), 4),
        "normalized_score": round(M.normalized_score(bf, start, ceiling), 4),
        "first_improvement_iter": M.first_improvement_iter(best_sf, start, iters),
        "iters_to_ceiling": M.iters_to_ceiling(best_sf, ceiling, iters),
        "success_rate": round(M.success_rate(best_sf, start), 4),
        # regression accuracy of the LLM's predicted LogS on the molecules it picked
        "pred_mae": round(reg["mae"], 4),
        "pred_rmse": round(reg["rmse"], 4),
        "pred_bias": round(reg["bias"], 4),
        "pred_pearson": round(reg["pearson"], 4),
        "pred_r2": round(reg["r2"], 4),
        "sign_acc_vs_seed": round(M.sign_accuracy(pred, true, start), 4),
        "n_evals": len(df),
    }


def report_metrics(runs, start, ceiling):
    rows = [run_metrics(a, r, df, start, ceiling)
            for (a, r), df in sorted(runs.items())]
    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "metrics2.csv")
    out.to_csv(path, index=False)
    print(f"\nApproach 2 per-run metrics (seed best = {start:+.4f}, "
          f"pool ceiling = {ceiling:+.4f}):")
    cols = ["acq", "run", "best_found", "simple_regret", "normalized_score",
            "success_rate", "pred_mae", "pred_rmse", "pred_r2", "sign_acc_vs_seed"]
    print(out[cols].to_string(index=False))
    print(f"Saved: {path}")


def main():
    runs = load_runs()
    if not runs:
        print("No Approach 2 result CSVs found in results/.")
        return
    print(f"Loaded {len(runs)} run(s): "
          + ", ".join(f"{a}{r}" for a, r in sorted(runs)))

    start = seed_best()
    ceiling, ceiling_name, vals = pool_ceiling()
    above = [v for v, _ in vals if v > start]

    print(f"\nseed best          = {start:+.4f}")
    print(f"pool ceiling       = {ceiling:+.4f}  ({ceiling_name})")
    print(f"pool molecules above seed best: {len(above)} / {len(vals)}")
    print("pool top 5 true ESOL LogS:")
    for v, n in vals[:5]:
        print(f"   {v:+.4f}  {n}")

    plot_convergence(runs, start, ceiling)
    mae, corr, bias = plot_calibration(runs)
    conf_err_corr = plot_confidence_error(runs)
    plot_comparison(runs, start, ceiling)

    # Global regression accuracy across every molecule any run actually picked.
    mu_all, true_all, _, _ = _all_points(runs)
    g = M.regression_metrics(mu_all, true_all)
    print(f"\nprediction MAE (|mu - true|)      = {g['mae']:.4f}")
    print(f"prediction RMSE                   = {g['rmse']:.4f}")
    print(f"prediction bias (mean mu - true)  = {g['bias']:+.4f}")
    print(f"corr(mu, true)  [Pearson]         = {g['pearson']:+.4f}")
    print(f"R^2 (mu explains true)            = {g['r2']:+.4f}")
    print(f"sign accuracy vs seed best        = "
          f"{M.sign_accuracy(mu_all, true_all, start):.4f}")
    print(f"corr(confidence, error)           = {conf_err_corr:+.4f}  "
          f"(negative = calibrated; near 0/positive = not)")

    report_metrics(runs, start, ceiling)
    print("Done.")


if __name__ == "__main__":
    main()
