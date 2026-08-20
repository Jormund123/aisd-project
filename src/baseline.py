"""
Random baseline for Approach 1.

Compares the LLM optimizer against naive random search on the SAME objective the
optimizer uses: ESOL-predicted LogS. For a budget of N evaluations we draw N
random AqSolDB molecules, score them with ESOL, and keep the best. Repeating this
K times gives the mean/std a random searcher would reach with the same budget.

We also report two ceilings over the whole AqSolDB:
  - best ESOL LogS   (the cap on the objective the optimizer actually maximizes)
  - best experimental LogS (the real-world cap, for reference)

Usage:
    python baseline.py --n 15 --k 1000
"""

import argparse
import os

import numpy as np
import pandas as pd

from esol import calculate_esol

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "aqsoldb.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_esol_scores():
    """Score every AqSolDB molecule with ESOL once. Drops molecules ESOL rejects."""
    df = pd.read_csv(DATA)
    scores = []
    for smi in df["SMILES"]:
        r = calculate_esol(smi)
        scores.append(r["logs_esol"] if r else np.nan)
    df["esol"] = scores
    return df.dropna(subset=["esol"]).reset_index(drop=True)


def random_baseline(esol_values, n, k, seed=0):
    """K independent trials: best ESOL LogS out of N random draws (no replacement)."""
    rng = np.random.default_rng(seed)
    bests = np.empty(k)
    for i in range(k):
        sample = rng.choice(esol_values, size=n, replace=False)
        bests[i] = sample.max()
    return bests


def main():
    parser = argparse.ArgumentParser(description="Random baseline vs LLM optimizer")
    parser.add_argument("--n", type=int, default=15, help="budget per trial")
    parser.add_argument("--k", type=int, default=1000, help="number of trials")
    args = parser.parse_args()

    print("Scoring AqSolDB with ESOL (one-time pass)...")
    df = load_esol_scores()
    esol_values = df["esol"].to_numpy()
    print(f"Scored {len(df)} molecules.")

    bests = random_baseline(esol_values, args.n, args.k)

    esol_ceiling_idx = df["esol"].idxmax()
    exp_ceiling_idx = df["Y"].idxmax()

    print("\n=== Random baseline (ESOL objective) ===")
    print(f"Budget N = {args.n}, trials K = {args.k}")
    print(f"Best-of-{args.n} LogS: mean {bests.mean():+.3f}  std {bests.std():.3f}")
    print(f"  range: [{bests.min():+.3f}, {bests.max():+.3f}]")

    print("\n=== Ceilings (whole AqSolDB) ===")
    print(f"Best ESOL:           {df.loc[esol_ceiling_idx, 'esol']:+.3f}  "
          f"{df.loc[esol_ceiling_idx, 'Name']}")
    print(f"Best experimental Y: {df.loc[exp_ceiling_idx, 'Y']:+.3f}  "
          f"{df.loc[exp_ceiling_idx, 'Name']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    trials_path = os.path.join(RESULTS_DIR, "baseline_trials.csv")
    pd.DataFrame({"trial": range(args.k), "best_of_n": bests}).to_csv(trials_path, index=False)

    summary_path = os.path.join(RESULTS_DIR, "baseline_summary.csv")
    pd.DataFrame([{
        "n": args.n,
        "k": args.k,
        "random_mean": bests.mean(),
        "random_std": bests.std(),
        "esol_ceiling": df.loc[esol_ceiling_idx, "esol"],
        "exp_ceiling": df.loc[exp_ceiling_idx, "Y"],
    }]).to_csv(summary_path, index=False)

    print(f"\nSaved: {trials_path}")
    print(f"       {summary_path}")


if __name__ == "__main__":
    main()
