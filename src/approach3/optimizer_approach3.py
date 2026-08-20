"""
Approach 3: Preferential Bayesian Optimization with the LLM as a pairwise ranker.

The LLM never predicts a number. Each round it judges a batch of duels
("which is more soluble, A or B?"). We aggregate all duels so far with a Bradley-Terry
model into a latent utility per candidate, then a preferential-BO acquisition
(UCB on the utility) picks ONE candidate to score with ESOL (the real oracle, counts
against budget). Repeat.

The LLM call stays manual: the script prints a duel prompt, the user pastes the JSON
of A/B answers back. An `--auto` mode replaces the human with an ESOL oracle (optionally
noisy) so the whole loop can be validated end to end without any LLM.

Usage:
    python optimizer_approach3.py --run 1 --budget 15            # manual (paste duels)
    python optimizer_approach3.py --auto --budget 15             # self-test, perfect ranker
    python optimizer_approach3.py --auto --noise 0.2 --budget 15 # self-test, noisy ranker
"""

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

# Shared modules live in src/ (..); Approach 2 code (reused here) lives in src/approach2.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "approach2"))

from esol import calculate_esol, validate_smiles
from seed_molecules import SEED_MOLECULES
from optimizer_approach2 import (
    build_seed_observed,
    build_pool,
    read_pasted_json,
    extract_json_array,
)
from prompt_templates3 import generate_pairwise_prompt
from pairwise import bradley_terry, pbo_acquisition

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "aqsoldb.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "approach3")

CSV_FIELDS = [
    "iteration", "run", "llm", "picked_id", "picked_smiles", "picked_name",
    "bt_utility", "bt_std_err", "acq_value", "n_duels_cumulative",
    "true_logs_esol", "best_so_far",
]
POOL_FIELDS = [
    "idx", "smiles", "name", "final_utility", "final_std_err",
    "true_logs_esol", "evaluated",
]


def build_stratified_pool(pool_size, seed, exclude_smiles, n_bins=6):
    """Sample a candidate pool spread across the ESOL range (not a flat random draw).

    AqSolDB is mostly insoluble molecules, so a random pool has almost nothing good to
    find (the Approach 2 failure mode: ceiling +0.21). The optimization target is ESOL,
    so we stratify by ESOL itself (NOT experimental Y, which the crude 4-descriptor ESOL
    formula does not track for salts/sugars). We compute ESOL for every valid,
    non-mixture, deduped molecule, split into n_bins equal-count (quantile) bins, and
    sample evenly across them. This guarantees the pool spans very-insoluble to soluble
    and has a ceiling meaningfully above the seed best, so ranking quality actually
    decides the outcome. Reproducible via seed.
    """
    df = pd.read_csv(DATA)
    seen = set(exclude_smiles)
    cans, names, esol = [], [], []
    for s, nm in zip(df["SMILES"], df.get("Name", [None] * len(df))):
        mol, canonical = validate_smiles(str(s))
        if mol is None or "." in canonical or canonical in seen:
            continue
        seen.add(canonical)
        cans.append(canonical)
        names.append(str(nm) if pd.notna(nm) else "")
        esol.append(calculate_esol(canonical)["logs_esol"])
    esol = np.asarray(esol)

    edges = np.quantile(esol, np.linspace(0.0, 1.0, n_bins + 1))
    bin_of = np.clip(np.digitize(esol, edges[1:-1]), 0, n_bins - 1)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cans))
    per_bin = int(np.ceil(pool_size / n_bins))
    counts = [0] * n_bins
    pool = []
    for idx in order:
        b = int(bin_of[idx])
        if counts[b] >= per_bin:
            continue
        pool.append({"name": names[idx], "smiles": cans[idx]})
        counts[b] += 1
        if len(pool) >= pool_size:
            break
    return pool[:pool_size]


def _key(a, b):
    return (a, b) if a < b else (b, a)


def select_duels(n, utility, std_err, evaluated, rng, n_duels):
    """Choose this round's duels (pairs of candidate indices to compare).

    Informative mix: neighbours in the current utility ranking (resolve close calls),
    the current leader vs everyone (confirm the top), and the most uncertain candidates
    vs the leader (explore). Ties (e.g. round 1, all-zero utility) are broken randomly
    so the first round covers the pool. Returns up to n_duels unique unordered pairs.
    """
    base = rng.permutation(n)
    order = base[np.argsort(-utility[base], kind="stable")]  # random tie-break
    leader = int(order[0])

    pairs = set()
    for a in range(n - 1):                       # neighbour pairs along the ranking
        pairs.add(_key(int(order[a]), int(order[a + 1])))
    others = [k for k in range(n) if k != leader]
    rng.shuffle(others)
    for k in others:                             # leader vs the field
        pairs.add(_key(leader, int(k)))
    for k in np.argsort(-std_err)[: max(3, n_duels // 3)]:  # most uncertain vs leader
        if int(k) != leader:
            pairs.add(_key(leader, int(k)))

    pairs = list(pairs)
    rng.shuffle(pairs)
    return pairs[:n_duels]


def oracle_answers(pairs, pool_true, noise, rng):
    """Auto-mode 'LLM': answer each duel by true ESOL, flipping with prob `noise`.

    Simulates a ranker of tunable quality (noise=0 is a perfect ranker). Returns a
    list of (i, j, winner) duels.
    """
    duels = []
    for (i, j) in pairs:
        win = i if pool_true[i] >= pool_true[j] else j
        if noise > 0 and rng.random() < noise:
            win = j if win == i else i
        duels.append((i, j, int(win)))
    return duels


def parse_pairwise(text, pairs):
    """Map a pasted JSON array of {q, winner} answers back onto duels.

    Returns (i, j, winner) for every answer that names a valid question and an A/B
    winner. Unanswered or malformed entries are simply dropped.
    """
    arr = extract_json_array(text)
    if not arr:
        return None
    duels = []
    for obj in arr:
        if not isinstance(obj, dict):
            continue
        try:
            q = int(obj.get("q"))
        except (TypeError, ValueError):
            continue
        if not (1 <= q <= len(pairs)):
            continue
        w = str(obj.get("winner", "")).strip().upper()
        i, j = pairs[q - 1]
        if w == "A":
            duels.append((i, j, i))
        elif w == "B":
            duels.append((i, j, j))
    return duels


def main():
    p = argparse.ArgumentParser(description="Approach 3: preferential BO, LLM ranker")
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--budget", type=int, default=15)
    p.add_argument("--llm", default="chatgpt")
    p.add_argument("--pool-size", type=int, default=30)
    p.add_argument("--pool-seed", type=int, default=0)
    p.add_argument("--n-bins", type=int, default=6)
    p.add_argument("--n-duels", type=int, default=25)
    p.add_argument("--kappa", type=float, default=2.0, help="UCB exploration weight")
    p.add_argument("--reg", type=float, default=1e-3, help="Bradley-Terry L2 ridge")
    p.add_argument("--random-pool", action="store_true",
                   help="use Approach 2's flat random pool instead of stratified")
    p.add_argument("--auto", action="store_true",
                   help="self-test: answer duels with an ESOL oracle, no LLM")
    p.add_argument("--noise", type=float, default=0.0,
                   help="auto-mode duel flip probability (ranker imperfection)")
    args = p.parse_args()

    observed = build_seed_observed()
    f_best = max(m["logs"] for m in observed)
    best = max(observed, key=lambda m: m["logs"])

    exclude = {m["smiles"] for m in observed}
    if args.random_pool:
        pool = build_pool(args.pool_size, args.pool_seed, exclude)
    else:
        pool = build_stratified_pool(args.pool_size, args.pool_seed, exclude, args.n_bins)
    n = len(pool)
    pool_true = np.array([calculate_esol(c["smiles"])["logs_esol"] for c in pool])

    evaluated = np.zeros(n, dtype=bool)
    duels = []
    utility = np.zeros(n)
    std_err = np.full(n, 1.0 / np.sqrt(args.reg))
    rng = np.random.default_rng(args.pool_seed + 1000 * args.run)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"approach3_run{args.run}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    pool_ceiling = float(pool_true.max())
    mode = f"AUTO(noise={args.noise})" if args.auto else f"MANUAL(llm={args.llm})"
    print(f"Approach 3 | Run {args.run} | Budget {args.budget} | {mode}")
    print(f"Pool {n} ({'random' if args.random_pool else 'stratified'}, seed "
          f"{args.pool_seed}) | pool ceiling = {pool_ceiling:+.3f}")
    print(f"Starting best (seed): {best['name']} LogS = {f_best:+.3f}")
    print(f"Logging to: {csv_path}")

    iteration = 1
    while iteration <= args.budget and not evaluated.all():
        pairs = select_duels(n, utility, std_err, evaluated, rng, args.n_duels)

        if args.auto:
            round_duels = oracle_answers(pairs, pool_true, args.noise, rng)
        else:
            print("\n" + "=" * 60)
            print(f"=== Iteration {iteration}/{args.budget} | {len(pairs)} duels ===")
            print(f"Best so far: {best['name']} LogS = {f_best:+.3f}")
            print("=" * 60)
            print("\n----- COPY THIS PROMPT -----\n")
            print(generate_pairwise_prompt(observed, pool, pairs))
            print("\n----- END PROMPT -----")
            round_duels = parse_pairwise(read_pasted_json(), pairs)
            if not round_duels:
                print("No usable answers parsed. Re-paste (does NOT use budget).")
                continue

        duels.extend(round_duels)
        utility, std_err = bradley_terry(duels, n, reg=args.reg)

        pick = pbo_acquisition(utility, std_err, evaluated, kappa=args.kappa)
        if pick is None:
            print("All candidates evaluated, stopping.")
            break

        cand = pool[pick]
        true_logs = float(pool_true[pick])
        evaluated[pick] = True
        acq_val = float(utility[pick] + args.kappa * std_err[pick])

        tag = ""
        if true_logs > f_best:
            f_best = true_logs
            best = {"name": cand["name"], "smiles": cand["smiles"], "logs": true_logs}
            tag = "  *** NEW BEST ***"
        print(f"[iter {iteration}] PICK #{pick + 1} {cand['smiles']} ({cand['name']}) "
              f"u={utility[pick]:+.3f} se={std_err[pick]:.3f} -> ESOL {true_logs:+.3f}{tag}")

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "iteration": iteration, "run": args.run,
                "llm": ("oracle" if args.auto else args.llm),
                "picked_id": pick + 1, "picked_smiles": cand["smiles"],
                "picked_name": cand["name"],
                "bt_utility": f"{utility[pick]:.4f}", "bt_std_err": f"{std_err[pick]:.4f}",
                "acq_value": f"{acq_val:.4f}", "n_duels_cumulative": len(duels),
                "true_logs_esol": f"{true_logs:.4f}", "best_so_far": f"{f_best:.4f}",
            })
        iteration += 1

    # Final full-pool ranking (for analysis3's ranking-quality figure).
    pool_path = os.path.join(RESULTS_DIR, f"approach3_run{args.run}_pool.csv")
    with open(pool_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POOL_FIELDS)
        w.writeheader()
        for k, c in enumerate(pool):
            w.writerow({
                "idx": k, "smiles": c["smiles"], "name": c["name"],
                "final_utility": f"{utility[k]:.4f}", "final_std_err": f"{std_err[k]:.4f}",
                "true_logs_esol": f"{pool_true[k]:.4f}", "evaluated": int(evaluated[k]),
            })

    print("\n" + "=" * 60)
    print(f"DONE. Best: {best['name']} {best['smiles']} LogS = {f_best:+.3f}")
    print(f"Pool ceiling was {pool_ceiling:+.3f}. Results: {csv_path}")
    print(f"Final pool ranking: {pool_path}")


if __name__ == "__main__":
    main()
