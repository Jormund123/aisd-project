"""
One-off setup script: samples molecules from the ZINC250k pool (data/zinc.tab,
the standard starting population for JNK3/GSK3B goal-directed generation
benchmarks, downloaded via tdc.generation.MolGen('ZINC')), scores them with the
JNK3 oracle, and writes data/jnk3_scored_pool.csv.

seed_molecules_jnk3.py then hand-picks a diverse few-shot set from this pool
(same role AqSolDB played for the ESOL seeds: source material, not the seeds
themselves). Only needs to run once.

Usage:
    python build_jnk3_seed_pool.py --n 3000
"""

import argparse
import csv
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from jnk3_oracle import calculate_jnk3_batch  # noqa: E402

ZINC_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "zinc.tab")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "jnk3_scored_pool.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000, help="molecules to sample and score")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = pd.read_csv(ZINC_PATH)
    sample = df["smiles"].sample(n=args.n, random_state=args.seed).tolist()

    print(f"Scoring {len(sample)} molecules from ZINC via JNK3 oracle...")
    results = calculate_jnk3_batch(sample)

    rows = [(r["canonical_smiles"], r["jnk3_score"]) for r in results if r is not None]
    rows.sort(key=lambda r: r[1], reverse=True)

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "jnk3_score"])
        w.writerows(rows)

    scores = [r[1] for r in rows]
    print(f"Scored {len(rows)}/{len(sample)} valid. "
          f"Range [{min(scores):.3f}, {max(scores):.3f}], "
          f"nonzero: {sum(1 for s in scores if s > 0)}")
    print(f"Top 10:")
    for smi, sc in rows[:10]:
        print(f"  {sc:.3f}  {smi}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
