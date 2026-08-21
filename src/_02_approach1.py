"""
Approach 1: LLM as Direct Optimizer. Interactive optimization loop, works for any
property registered in common.PROPERTIES.

The LLM call stays manual (no API): the script prints a prompt, the user pastes it
into ChatGPT/Claude/Gemini, then pastes the JSON response back. The script handles
everything else: validation, scoring, duplicate checking, best-so-far tracking, CSV
logging.

Usage:
    python _02_approach1.py --property esol --variant B --run 1 --budget 15 --llm gemini
    python _02_approach1.py --property jnk3 --variant B --run 1 --budget 15 --llm gemini
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import PROPERTIES, validate_smiles, read_pasted_json, extract_json  # noqa: E402
from _01_seed_molecules import get_seeds  # noqa: E402

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "results", "approach1")

CSV_FIELDS = [
    "iteration", "variant", "llm", "smiles_proposed", "name",
    "valid", "duplicate", "canonical_smiles", "score",
    "best_so_far", "reasoning",
]

JSON_BLOCK = """{
  "smiles": "<valid SMILES string>",
  "name": "<common or IUPAC name>",
  "reasoning": "<reasoning>"
}"""

REQUIREMENTS = (
    "Requirements:\n"
    "- Must be a real, synthesizable organic compound\n"
    "- Do not repeat any molecule already listed\n"
    "- Respond ONLY with valid JSON, no other text\n\n"
)


def _format_known(known, label):
    rows = sorted(known, key=lambda m: m["score"], reverse=True)
    lines = []
    for i, m in enumerate(rows, 1):
        lines.append(f"{i:>2}. {m['name']:<14} {m['smiles']:<40} {label} = {m['score']:.3f}")
    return "\n".join(lines)


def generate_prompt(known, variant, property_name):
    prop = PROPERTIES[property_name]
    header = (
        f"You are a computational chemist optimizing predicted {prop['prompt_property_desc']}.\n\n"
        + prop["domain_hint"]
        + f"Below are molecules that have been tested, with their {prop['label']} values:\n\n"
    )
    table = _format_known(known, prop["label"])
    body = header + table + "\n\n"

    step1 = (
        f"Step 1: Analyze the data above. Which structural features correlate with "
        f"HIGH {prop['label']}? Which correlate with LOW {prop['label']}?\n\n"
        f"Step 2: Using that analysis, design ONE new molecule predicted to have a "
        f"HIGHER {prop['label']} than any molecule above"
    )

    if variant == "A":
        return body + step1 + ".\n\n" + REQUIREMENTS + JSON_BLOCK

    if variant == "B":
        instruction = (
            step1 + ".\n\n"
            "Think carefully and avoid trivial solutions:\n"
            "- Do NOT simply extend a homologous series or repeat one motif. That "
            "exploits the scoring function but is not a real discovery.\n"
            "- Propose a structurally distinct, realistic molecule that a practicing "
            "chemist would consider genuinely synthesizable and useful.\n"
            "- Favor chemical realism over blindly maximizing one descriptor.\n\n"
        )
        return body + instruction + REQUIREMENTS + JSON_BLOCK

    if variant == "C":
        instruction = (
            step1 + ", obeying every constraint below.\n\n"
            "Hard constraints:\n"
            + prop["variant_c_constraints"] +
            "- Must be a SINGLE, connected molecule: no salts, mixtures, co-crystals, "
            "or disconnected components (no '.' in the SMILES)\n"
            "- Must be a real, synthesizable organic compound\n"
            "- Do not repeat any molecule already listed\n\n"
            "Think carefully and avoid trivial solutions:\n"
            "- Do NOT simply extend a homologous series or repeat one motif. That "
            "exploits the scoring function but is not a real discovery.\n"
            "- Favor chemical realism over blindly maximizing one descriptor.\n\n"
            "Respond ONLY with valid JSON, no other text:\n\n"
        )
        return body + instruction + JSON_BLOCK

    raise ValueError(f"Unknown variant: {variant} (expected A, B, or C)")


def load_existing(csv_path):
    rows, added = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if row.get("valid") == "True" and row.get("duplicate") == "False":
                added.append({"name": row["name"] or row["canonical_smiles"],
                              "smiles": row["canonical_smiles"], "score": float(row["score"])})
    next_iter = (max(int(r["iteration"]) for r in rows) + 1) if rows else 1
    return rows, added, next_iter


def main():
    parser = argparse.ArgumentParser(description="Approach 1: LLM as Direct Optimizer")
    parser.add_argument("--property", default="esol", choices=list(PROPERTIES))
    parser.add_argument("--variant", default="B", choices=["A", "B", "C"])
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--llm", default="gemini")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an existing CSV instead of overwriting it.")
    args = parser.parse_args()

    prop = PROPERTIES[args.property]
    known = get_seeds(args.property)
    tested_smiles = {m["smiles"] for m in known}

    results_dir = RESULTS_ROOT if args.property == "esol" else os.path.join(RESULTS_ROOT, args.property)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"approach1_variant{args.variant}_run{args.run}.csv")

    if args.resume and os.path.exists(csv_path):
        _, added, iteration = load_existing(csv_path)
        for m in added:
            known.append(m)
            tested_smiles.add(m["smiles"])
        best = max(known, key=lambda m: m["score"])
        print(f"RESUMING from iteration {iteration}. Loaded {len(added)} prior molecule(s).")
    else:
        iteration = 1
        best = max(known, key=lambda m: m["score"])
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    print(f"Approach 1 ({args.property}) | Variant {args.variant} | Run {args.run} | "
          f"Budget {args.budget} | LLM {args.llm}")
    print(f"Logging to: {csv_path}")
    print(f"Starting best: {best['name']} {prop['label']} = {best['score']:.3f}")

    while iteration <= args.budget:
        print("\n" + "=" * 60)
        print(f"=== Iteration {iteration}/{args.budget}  (Variant {args.variant}) ===")
        print(f"Best so far: {best['name']}  {prop['label']} = {best['score']:.3f}")
        print("=" * 60)
        print("\n----- COPY THIS PROMPT -----\n")
        print(generate_prompt(known, args.variant, args.property))
        print("\n----- END PROMPT -----")

        data = extract_json(read_pasted_json())
        if data is None or "smiles" not in data:
            print("Could not parse JSON or no 'smiles' field. Re-paste (does NOT use budget).")
            continue

        smiles = str(data.get("smiles", "")).strip()
        name = str(data.get("name", "")).strip()
        reasoning = str(data.get("reasoning", "")).strip()

        mol, canonical = validate_smiles(smiles)
        valid = mol is not None
        if valid and "." in canonical:
            valid = False
            print(f"REJECTED MIXTURE (disconnected SMILES): {canonical}")
        duplicate = valid and canonical in tested_smiles
        score = None

        if not valid:
            print(f"INVALID SMILES: {smiles}  -> failed iteration (counts against budget).")
        elif duplicate:
            print(f"DUPLICATE: {canonical} already tested -> failed iteration "
                  "(counts against budget).")
        else:
            score = prop["score"](smiles)
            known.append({"name": name or canonical, "smiles": canonical, "score": score})
            tested_smiles.add(canonical)
            if score > best["score"]:
                best = {"name": name or canonical, "smiles": canonical, "score": score}
                print(f"VALID: {canonical}  {prop['label']} = {score:.3f}  *** NEW BEST ***")
            else:
                print(f"VALID: {canonical}  {prop['label']} = {score:.3f}  "
                      f"(best stays {best['score']:.3f})")

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "iteration": iteration, "variant": args.variant, "llm": args.llm,
                "smiles_proposed": smiles, "name": name, "valid": valid, "duplicate": duplicate,
                "canonical_smiles": canonical if valid else "",
                "score": f"{score:.4f}" if score is not None else "",
                "best_so_far": f"{best['score']:.4f}", "reasoning": reasoning,
            })
        iteration += 1

    print("\n" + "=" * 60)
    print(f"DONE. {args.budget} iterations complete.")
    print(f"Best: {best['name']}  {best['smiles']}  {prop['label']} = {best['score']:.3f}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
