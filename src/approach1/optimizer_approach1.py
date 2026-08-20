"""
Approach 1: LLM as Direct Optimizer. Interactive optimization loop.

The LLM call stays manual (no API): the script prints a prompt, the user pastes
it into ChatGPT/Claude/Gemini, then pastes the JSON response back. The script
handles everything else: validation, ESOL scoring, duplicate checking, best-so-far
tracking, and CSV logging.

Usage:
    python optimizer_approach1.py --variant B --run 1 --budget 15 --llm gemini
"""

import argparse
import csv
import json
import os
import re
import sys

# Shared modules (esol, seed_molecules) live one level up in src/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from esol import calculate_esol, validate_smiles
from seed_molecules import SEED_MOLECULES
from prompt_templates import generate_prompt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "approach1")

CSV_FIELDS = [
    "iteration", "variant", "llm", "smiles_proposed", "name",
    "valid", "duplicate", "canonical_smiles", "logs_esol",
    "best_so_far", "reasoning",
]


def build_seed_known():
    """Seed molecules scored by ESOL (consistent with the prompt scoreboard)."""
    known = []
    for m in SEED_MOLECULES:
        r = calculate_esol(m["smiles"])
        known.append({
            "name": m["name"],
            "smiles": r["canonical_smiles"],
            "logs": r["logs_esol"],
        })
    return known


def read_pasted_json():
    """Read a pasted multi-line response, terminated by an empty line."""
    print("\nPaste LLM JSON response, then press Enter on an empty line:")
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


def extract_json(text):
    """Pull the first {...} block out of pasted text and parse it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def load_existing(csv_path):
    """Read a partial run's CSV. Returns (rows, added_molecules, next_iteration).
    added_molecules = valid, non-duplicate proposals already logged."""
    rows = []
    added = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if row.get("valid") == "True" and row.get("duplicate") == "False":
                added.append({
                    "name": row["name"] or row["canonical_smiles"],
                    "smiles": row["canonical_smiles"],
                    "logs": float(row["logs_esol"]),
                })
    next_iter = (max(int(r["iteration"]) for r in rows) + 1) if rows else 1
    return rows, added, next_iter


def main():
    parser = argparse.ArgumentParser(description="Approach 1: LLM as Direct Optimizer")
    parser.add_argument("--variant", default="B", choices=["A", "B", "C"])
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--llm", default="gemini")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an existing CSV instead of overwriting it.")
    args = parser.parse_args()

    known = build_seed_known()
    tested_smiles = {m["smiles"] for m in known}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(
        RESULTS_DIR, f"approach1_variant{args.variant}_run{args.run}.csv"
    )

    if args.resume and os.path.exists(csv_path):
        _, added, iteration = load_existing(csv_path)
        for m in added:
            known.append(m)
            tested_smiles.add(m["smiles"])
        best = max(known, key=lambda m: m["logs"])
        print(f"RESUMING from iteration {iteration}. "
              f"Loaded {len(added)} prior molecule(s).")
    else:
        iteration = 1
        best = max(known, key=lambda m: m["logs"])
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    print(f"Approach 1 | Variant {args.variant} | Run {args.run} | "
          f"Budget {args.budget} | LLM {args.llm}")
    print(f"Logging to: {csv_path}")
    print(f"Starting best: {best['name']} LogS = {best['logs']:+.3f}")
    while iteration <= args.budget:
        print("\n" + "=" * 60)
        print(f"=== Iteration {iteration}/{args.budget}  (Variant {args.variant}) ===")
        print(f"Best so far: {best['name']}  LogS = {best['logs']:+.3f}")
        print("=" * 60)
        print("\n----- COPY THIS PROMPT -----\n")
        print(generate_prompt(known, args.variant))
        print("\n----- END PROMPT -----")

        data = extract_json(read_pasted_json())
        if data is None or "smiles" not in data:
            print("Could not parse JSON or no 'smiles' field. "
                  "Re-paste (does NOT use budget).")
            continue

        smiles = str(data.get("smiles", "")).strip()
        name = str(data.get("name", "")).strip()
        reasoning = str(data.get("reasoning", "")).strip()

        mol, canonical = validate_smiles(smiles)
        valid = mol is not None
        # Reject salts/mixtures/co-crystals: ESOL on disconnected fragments just
        # sums them, which exploits the objective (see Variant B iter 10).
        if valid and "." in canonical:
            valid = False
            print(f"REJECTED MIXTURE (disconnected SMILES): {canonical}")
        duplicate = valid and canonical in tested_smiles
        logs = None

        if not valid:
            print(f"INVALID SMILES: {smiles}  -> failed iteration (counts against budget).")
        elif duplicate:
            print(f"DUPLICATE: {canonical} already tested -> failed iteration "
                  "(counts against budget).")
        else:
            logs = calculate_esol(smiles)["logs_esol"]
            known.append({"name": name or canonical, "smiles": canonical, "logs": logs})
            tested_smiles.add(canonical)
            if logs > best["logs"]:
                best = {"name": name or canonical, "smiles": canonical, "logs": logs}
                print(f"VALID: {canonical}  LogS = {logs:+.3f}  *** NEW BEST ***")
            else:
                print(f"VALID: {canonical}  LogS = {logs:+.3f}  "
                      f"(best stays {best['logs']:+.3f})")

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "iteration": iteration,
                "variant": args.variant,
                "llm": args.llm,
                "smiles_proposed": smiles,
                "name": name,
                "valid": valid,
                "duplicate": duplicate,
                "canonical_smiles": canonical if valid else "",
                "logs_esol": f"{logs:.4f}" if logs is not None else "",
                "best_so_far": f"{best['logs']:.4f}",
                "reasoning": reasoning,
            })
        iteration += 1

    print("\n" + "=" * 60)
    print(f"DONE. {args.budget} iterations complete.")
    print(f"Best: {best['name']}  {best['smiles']}  LogS = {best['logs']:+.3f}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
