"""
Metrics for all three approaches, any property. Also folds in the random-baseline
comparison (was baseline.py).

Column note: runs collected before this refactor (the real, historical, hand-pasted
ESOL manual-LLM sessions -- can't be regenerated) used property-specific column names
(logs_esol, true_logs_esol). Runs from _02/_03/_04 (both properties, going forward)
use property-neutral names (score, true_score). _alias_columns() below makes both
readable through one code path without touching the historical CSVs.

Usage:
    python _05_analysis.py --approach 1 --property esol
    python _05_analysis.py --approach 2 --property jnk3
    python _05_analysis.py --approach 3 --property esol
    python _05_analysis.py --approach baseline --property jnk3 --n 15 --k 1000
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
from _01_seed_molecules import get_seeds  # noqa: E402

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "results")

_LEGACY_ALIASES = {
    "logs_esol": "score", "true_logs_esol": "true_score",
    "predicted_logs": "predicted_score",
}


def _alias_columns(df):
    return df.rename(columns={k: v for k, v in _LEGACY_ALIASES.items() if k in df.columns})


def _results_dir(approach, property_name):
    base = os.path.join(RESULTS_ROOT, f"approach{approach}")
    return base if property_name == "esol" else os.path.join(base, property_name)


def _is_valid(series):
    return series.astype(str).str.strip().str.lower() == "true"


# --------------------------------------------------------------------------- #
# Loaders (reused by _06_plots.py)
# --------------------------------------------------------------------------- #

def load_runs_1(property_name):
    """{(variant, run): DataFrame} for every Approach 1 result CSV."""
    d = _results_dir(1, property_name)
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "approach1_variant*_run*.csv"))):
        m = re.search(r"variant([A-C])_run(\d+)", os.path.basename(f))
        if not m:
            continue
        df = _alias_columns(pd.read_csv(f))
        if len(df):
            runs[(m.group(1), int(m.group(2)))] = df
    return runs


def load_runs_2(property_name):
    """{(acq, run): DataFrame} for every Approach 2 generative result CSV."""
    d = _results_dir(2, property_name)
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "approach2gen_*_run*.csv"))):
        m = re.search(r"approach2gen_(ei|ucb)_run(\d+)", os.path.basename(f))
        if not m:
            continue
        df = _alias_columns(pd.read_csv(f))
        if len(df):
            runs[(m.group(1), int(m.group(2)))] = df
    return runs


def load_runs_3(property_name):
    """{run: (opt_df, ranking_df)} for every Approach 3 generative result CSV."""
    d = _results_dir(3, property_name)
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "approach3gen_run*.csv"))):
        base = os.path.basename(f)
        if "ranking" in base:
            continue
        m = re.search(r"approach3gen_run(\d+)\.csv", base)
        if not m:
            continue
        run = int(m.group(1))
        opt = _alias_columns(pd.read_csv(f))
        rank_path = os.path.join(d, f"approach3gen_run{run}_ranking.csv")
        rank = _alias_columns(pd.read_csv(rank_path)) if os.path.exists(rank_path) else None
        if len(opt):
            runs[run] = (opt, rank)
    return runs


def _mean_pairwise_similarity(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    if len(fps) < 2:
        return float("nan")
    sims = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    return float(np.mean(sims)) if sims else float("nan")


# --------------------------------------------------------------------------- #
# Approach 1
# --------------------------------------------------------------------------- #

def analyze_approach1(property_name):
    runs = load_runs_1(property_name)
    if not runs:
        print("No Approach 1 result CSVs found.")
        return
    start = max(m["score"] for m in get_seeds(property_name))

    rows = []
    for (variant, run), df in sorted(runs.items()):
        best_sf = df["best_so_far"].astype(float).to_numpy()
        iters = df["iteration"].astype(int).tolist()
        valid = _is_valid(df["valid"])
        true_vals = df.loc[valid, "score"].astype(float).to_numpy()
        smiles = df.loc[valid, "canonical_smiles"].dropna().tolist()
        bf = C.best_found(best_sf, start)
        rows.append({
            "approach": 1, "variant": variant, "run": run,
            "best_found": round(bf, 4),
            "improvement_over_seed": round(C.improvement_over_seed(bf, start), 4),
            "first_improvement_iter": C.first_improvement_iter(best_sf, start, iters),
            "success_rate": round(C.success_rate(best_sf, start), 4),
            "hit_rate": round(C.hit_rate(true_vals, start), 4),
            "invalid_rate": round(1 - valid.mean(), 4),
            "diversity_tanimoto": round(_mean_pairwise_similarity(smiles), 4),
            "n_evals": len(df),
        })
    out = pd.DataFrame(rows)
    path = os.path.join(_results_dir(1, property_name), "metrics1.csv")
    out.to_csv(path, index=False)
    print(f"Approach 1 ({property_name}) per-run metrics (seed best = {start:+.4f}):")
    cols = ["variant", "run", "best_found", "improvement_over_seed", "success_rate",
            "hit_rate", "invalid_rate", "diversity_tanimoto", "first_improvement_iter"]
    print(out[cols].to_string(index=False))
    print(f"Saved: {path}")


# --------------------------------------------------------------------------- #
# Approach 2 (generative)
# --------------------------------------------------------------------------- #

def analyze_approach2(property_name):
    runs = load_runs_2(property_name)
    if not runs:
        print("No Approach 2 generative result CSVs found.")
        return
    start = max(m["score"] for m in get_seeds(property_name))

    rows = []
    for (acq, run), df in sorted(runs.items()):
        d = df.dropna(subset=["surrogate_label"])
        pred = d["surrogate_label"].astype(float).to_numpy()
        true = d["true_score"].astype(float).to_numpy()
        err = C.regression_error(pred, true)
        rows.append({
            "acq": acq, "run": run,
            "best_found": round(float(df["best_so_far"].astype(float).max()), 4),
            "improvement_over_seed":
                round(float(df["best_so_far"].astype(float).max()) - start, 4),
            "pairwise_acc": round(C.pairwise_accuracy(pred, true), 4),
            "pred_mae": round(err["mae"], 4),
            "pred_rmse": round(err["rmse"], 4),
            "pred_bias": round(err["bias"], 4),
            "n_evals": len(df),
        })
    out = pd.DataFrame(rows)
    path = os.path.join(_results_dir(2, property_name), "metrics2_gen.csv")
    out.to_csv(path, index=False)

    all_pred, all_true = [], []
    for (_, _), df in runs.items():
        d = df.dropna(subset=["surrogate_label"])
        all_pred.extend(d["surrogate_label"].astype(float).tolist())
        all_true.extend(d["true_score"].astype(float).tolist())
    print(f"Approach 2 ({property_name}) | seed best = {start:+.4f}")
    print(out.to_string(index=False))
    print(f"Pooled pairwise accuracy = {C.pairwise_accuracy(all_pred, all_true):.4f}")
    print(f"Saved: {path}")


# --------------------------------------------------------------------------- #
# Approach 3 (generative)
# --------------------------------------------------------------------------- #

def analyze_approach3(property_name):
    runs = load_runs_3(property_name)
    if not runs:
        print("No Approach 3 generative result CSVs found.")
        return
    start = max(m["score"] for m in get_seeds(property_name))

    rows = []
    for run, (opt, rank) in sorted(runs.items()):
        pacc = float("nan")
        if rank is not None:
            pacc = C.pairwise_accuracy(rank["bt_utility"], rank["true_score"])
        rows.append({
            "run": run,
            "best_found": round(float(opt["best_so_far"].astype(float).max()), 4),
            "improvement_over_seed":
                round(float(opt["best_so_far"].astype(float).max()) - start, 4),
            "pairwise_acc": round(pacc, 4),
            "n_duels": int(opt["n_duels_cum"].astype(int).max()),
            "n_evals": len(opt),
        })
    out = pd.DataFrame(rows)
    path = os.path.join(_results_dir(3, property_name), "metrics3_gen.csv")
    out.to_csv(path, index=False)
    print(f"Approach 3 ({property_name}) | seed best = {start:+.4f}")
    print(out.to_string(index=False))
    print(f"Saved: {path}")


# --------------------------------------------------------------------------- #
# Baseline (was baseline.py)
# --------------------------------------------------------------------------- #

def analyze_baseline(property_name, n, k):
    mean, std, ceiling = C.run_baseline(property_name, n=n, k=k)
    start = max(m["score"] for m in get_seeds(property_name))
    print(f"=== Random baseline ({property_name}) ===")
    print(f"Budget N = {n}, trials K = {k}")
    print(f"Best-of-{n}: mean {mean:+.3f}  std {std:.3f}")
    print(f"Pool ceiling: {ceiling:+.3f}  |  seed best: {start:+.3f}")

    name = "baseline_summary.csv" if property_name == "esol" else f"baseline_summary_{property_name}.csv"
    path = os.path.join(RESULTS_ROOT, name)
    pd.DataFrame([{
        "property": property_name, "n": n, "k": k,
        "random_mean": mean, "random_std": std, "ceiling": ceiling, "seed_best": start,
    }]).to_csv(path, index=False)
    print(f"Saved: {path}")


def main():
    p = argparse.ArgumentParser(description="Metrics for all approaches, any property")
    p.add_argument("--approach", required=True, choices=["1", "2", "3", "baseline"])
    p.add_argument("--property", default="esol", choices=list(C.PROPERTIES))
    p.add_argument("--n", type=int, default=15, help="baseline: budget per trial")
    p.add_argument("--k", type=int, default=1000, help="baseline: number of trials")
    args = p.parse_args()

    if args.approach == "1":
        analyze_approach1(args.property)
    elif args.approach == "2":
        analyze_approach2(args.property)
    elif args.approach == "3":
        analyze_approach3(args.property)
    elif args.approach == "baseline":
        analyze_baseline(args.property, args.n, args.k)


if __name__ == "__main__":
    main()
