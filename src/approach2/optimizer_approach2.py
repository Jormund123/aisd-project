"""
Approach 2: Bayesian Optimization with the LLM as a regressor.

The LLM no longer designs molecules. It predicts LogS + confidence for a FIXED pool
of candidate molecules (sampled from AqSolDB). An acquisition function (EI or UCB)
turns those predictions into a single pick, which we then score with ESOL (the real
oracle, counts against budget). Repeat until budget is spent.

LLM call stays manual: the script prints a prediction prompt, the user pastes it into
ChatGPT/Claude/Gemini, then pastes the JSON array of predictions back. The script
does pool sampling, acquisition math, ESOL scoring, and CSV logging.

Usage:
    python optimizer_approach2.py --run 1 --acq ei --budget 15 --pool-size 30
"""

import argparse
import csv
import json
import os
import re

import sys

import numpy as np
import pandas as pd

# Shared modules (esol, seed_molecules) live one level up in src/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from esol import calculate_esol, validate_smiles
from seed_molecules import SEED_MOLECULES
from prompt_templates2 import generate_regression_prompt
from acquisition import score_candidates

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "aqsoldb.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "approach2")

CSV_FIELDS = [
    "iteration", "run", "llm", "acq", "picked_id", "picked_smiles", "picked_name",
    "predicted_logs", "confidence", "sigma", "acq_value",
    "true_logs_esol", "best_so_far", "n_scored",
]


def build_seed_observed():
    """Seed molecules scored by ESOL (true oracle values the LLM gets to see)."""
    observed = []
    for m in SEED_MOLECULES:
        r = calculate_esol(m["smiles"])
        observed.append({"name": m["name"], "smiles": r["canonical_smiles"],
                         "logs": r["logs_esol"]})
    return observed


def build_pool(pool_size, seed, exclude_smiles):
    """Sample a fixed candidate pool from AqSolDB (canonical, valid, deduped).

    Candidates carry only name + SMILES. Their LogS is hidden from the LLM and is
    revealed (via ESOL) only when the acquisition function picks them.
    """
    df = pd.read_csv(DATA)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    pool = []
    seen = set(exclude_smiles)
    for idx in order:
        row = df.iloc[idx]
        mol, canonical = validate_smiles(str(row["SMILES"]))
        if mol is None or "." in canonical or canonical in seen:
            continue
        seen.add(canonical)
        name = str(row["Name"]) if "Name" in df.columns and pd.notna(row["Name"]) else ""
        pool.append({"name": name, "smiles": canonical})
        if len(pool) >= pool_size:
            break
    return pool


def read_pasted_json():
    """Read a pasted multi-line response, terminated by an empty line."""
    print("\nPaste LLM JSON array of predictions, then Enter on an empty line:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            if lines:
                break
            continue
        lines.append(line)
    return "\n".join(lines)


def extract_json_array(text):
    """Pull the first [...] block out of pasted text and parse it to a list."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def match_predictions(preds, candidates):
    """Align LLM predictions to remaining candidates.

    Match by 1-based id first, then by canonical SMILES. Returns parallel lists
    (idx_in_candidates, mu, confidence) for every candidate that got a usable
    prediction. Candidates with no prediction are simply not eligible this round.
    """
    by_smiles = {}
    for j, c in enumerate(candidates):
        by_smiles[c["smiles"]] = j

    idxs, mus, confs = [], [], []
    used = set()
    for p in preds:
        if not isinstance(p, dict) or "predicted_logs" not in p:
            continue
        j = None
        pid = p.get("id")
        if isinstance(pid, int) and 1 <= pid <= len(candidates):
            j = pid - 1
        if j is None and "smiles" in p:
            _, canon = validate_smiles(str(p["smiles"]))
            if canon in by_smiles:
                j = by_smiles[canon]
        if j is None or j in used:
            continue
        try:
            mu = float(p["predicted_logs"])
            conf = float(p.get("confidence", 5))
        except (TypeError, ValueError):
            continue
        used.add(j)
        idxs.append(j)
        mus.append(mu)
        confs.append(conf)
    return idxs, mus, confs


def main():
    parser = argparse.ArgumentParser(description="Approach 2: BO, LLM as regressor")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--acq", default="ei", choices=["ei", "ucb"])
    parser.add_argument("--llm", default="gemini")
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--pool-seed", type=int, default=0,
                        help="RNG seed for the AqSolDB candidate pool (reproducible).")
    parser.add_argument("--kappa", type=float, default=2.0, help="UCB exploration weight")
    parser.add_argument("--xi", type=float, default=0.0, help="EI exploration margin")
    parser.add_argument("--sigma-min", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    args = parser.parse_args()

    observed = build_seed_observed()
    f_best = max(m["logs"] for m in observed)
    best = max(observed, key=lambda m: m["logs"])

    pool = build_pool(args.pool_size, args.pool_seed,
                      exclude_smiles={m["smiles"] for m in observed})
    evaluated = set()  # canonical SMILES already tested by ESOL

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"approach2_{args.acq}_run{args.run}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    print(f"Approach 2 | Acq {args.acq.upper()} | Run {args.run} | Budget {args.budget} "
          f"| Pool {len(pool)} (seed {args.pool_seed}) | LLM {args.llm}")
    print(f"Logging to: {csv_path}")
    print(f"Starting best (seed): {best['name']} LogS = {best['logs']:+.3f}")

    iteration = 1
    while iteration <= args.budget:
        candidates = [c for c in pool if c["smiles"] not in evaluated]
        if not candidates:
            print("Pool exhausted, stopping early.")
            break

        print("\n" + "=" * 60)
        print(f"=== Iteration {iteration}/{args.budget}  (acq {args.acq.upper()}) ===")
        print(f"Best so far: {best['name']}  LogS = {best['logs']:+.3f} "
              f"| candidates left: {len(candidates)}")
        print("=" * 60)
        print("\n----- COPY THIS PROMPT -----\n")
        print(generate_regression_prompt(observed, candidates))
        print("\n----- END PROMPT -----")

        preds = extract_json_array(read_pasted_json())
        if not preds:
            print("Could not parse a JSON array. Re-paste (does NOT use budget).")
            continue

        idxs, mus, confs = match_predictions(preds, candidates)
        if not idxs:
            print("No prediction matched a candidate. Re-paste (does NOT use budget).")
            continue

        scores, sigmas = score_candidates(
            mus, confs, f_best, acq=args.acq,
            kappa=args.kappa, xi=args.xi,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        )
        winner = int(np.argmax(scores))
        cand = candidates[idxs[winner]]
        mu_w, conf_w = mus[winner], confs[winner]
        sigma_w, acq_w = float(sigmas[winner]), float(scores[winner])

        true_logs = calculate_esol(cand["smiles"])["logs_esol"]
        evaluated.add(cand["smiles"])
        observed.append({"name": cand["name"], "smiles": cand["smiles"], "logs": true_logs})

        tag = ""
        if true_logs > f_best:
            f_best = true_logs
            best = {"name": cand["name"], "smiles": cand["smiles"], "logs": true_logs}
            tag = "  *** NEW BEST ***"
        print(f"PICKED #{idxs[winner] + 1} {cand['smiles']} ({cand['name']})")
        print(f"  LLM mu={mu_w:+.3f} conf={conf_w:.0f} sigma={sigma_w:.3f} "
              f"acq={acq_w:.4f} -> ESOL LogS={true_logs:+.3f}{tag}")

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "iteration": iteration, "run": args.run, "llm": args.llm,
                "acq": args.acq, "picked_id": idxs[winner] + 1,
                "picked_smiles": cand["smiles"], "picked_name": cand["name"],
                "predicted_logs": f"{mu_w:.4f}", "confidence": f"{conf_w:.0f}",
                "sigma": f"{sigma_w:.4f}", "acq_value": f"{acq_w:.4f}",
                "true_logs_esol": f"{true_logs:.4f}", "best_so_far": f"{f_best:.4f}",
                "n_scored": len(idxs),
            })
        iteration += 1

    print("\n" + "=" * 60)
    print(f"DONE. Best: {best['name']}  {best['smiles']}  LogS = {best['logs']:+.3f}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
