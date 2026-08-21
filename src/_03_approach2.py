"""
Approach 2 (generative): Bayesian Optimization with the LLM as a regressor, searching
the SELFIES-VAE latent space. Works for any property registered in common.PROPERTIES.

Per round:
  1. Fit a GP on the latent vectors of the labeled molecules and their surrogate labels
     (LLM-predicted score; in --auto the property's own oracle stands in for the LLM).
  2. Generate NEW candidate molecules by sampling latents near the known-good region and
     decoding them with the VAE.
  3. GP posterior (mean carries the LLM signal, std grows away from known data) feeds the
     EI/UCB acquisition -> pick a small batch.
  4. Score the picks with the property's oracle (counts against budget) for the
     convergence curve, then ask the LLM to predict just those picks and add them to
     the labeled set.

Manual LLM stays copy-paste: the script prints a prediction prompt for the picked
molecules and reads back a JSON array. --auto replaces that with the oracle so the
whole loop can be validated end to end with no LLM.

Usage:
    python _03_approach2.py --property esol --run 1 --acq ei --budget 15 --batch 5
    python _03_approach2.py --property jnk3 --auto --budget 15          # validation
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    PROPERTIES, SelfiesVAE, GP, generate_candidates, rank_candidates,
    read_pasted_json, extract_json_array, match_predictions,
)
from _01_seed_molecules import get_seeds  # noqa: E402

VAE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "selfies_vae.pt")
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "results", "approach2")

CSV_FIELDS = [
    "eval_idx", "round", "run", "llm", "acq", "picked_smiles",
    "gp_mu", "gp_sigma", "acq_value", "surrogate_label",
    "true_score", "best_so_far",
]


def generate_regression_prompt(observed, candidates, prop):
    lines_obs = [f"{i:2d}. {m['smiles']}: {prop['label']} = {m['score']:.3f}"
                 for i, m in enumerate(observed, 1)]
    lines_cand = [f"{i:2d}. {c['smiles']}" for i, c in enumerate(candidates, 1)]
    return f"""You are a computational chemist acting as a prediction model for {prop['prompt_property_desc']}.

{prop['domain_hint']}Below are molecules already measured, with their {prop['label']} values:

{chr(10).join(lines_obs)}

Now predict the {prop['label']} for EACH candidate molecule below. For each one give:
  - predicted_score: your best numeric estimate ({prop['label']})
  - confidence: an integer 1-10 (10 = very sure, 1 = wild guess)

Candidates:
{chr(10).join(lines_cand)}

Respond ONLY with a JSON array as snippet code, one object per candidate, no other text:

[
  {{"id": 1, "smiles": "<candidate SMILES>", "predicted_score": <number>, "confidence": <1-10>}},
  {{"id": 2, "smiles": "<candidate SMILES>", "predicted_score": <number>, "confidence": <1-10>}}
]

Predict every candidate. Keep the same id and smiles shown above.
"""


def label_picks_auto(smiles_list, prop, rng, noise):
    scores = prop["score_batch"](smiles_list)
    labels = []
    for s in scores:
        y = s if s is not None else 0.0
        if noise > 0:
            y += rng.normal(0.0, noise)
        labels.append(y)
    return labels


def label_picks_manual(context, smiles_list, prop):
    candidates = [{"name": "", "smiles": s} for s in smiles_list]
    print("\n----- COPY THIS PROMPT -----\n")
    print(generate_regression_prompt(context, candidates, prop))
    print("\n----- END PROMPT -----")
    preds = extract_json_array(read_pasted_json())
    labels = [None] * len(smiles_list)
    if not preds:
        return labels
    idxs, mus, _ = match_predictions(preds, candidates)
    for j, mu in zip(idxs, mus):
        labels[j] = mu
    return labels


def main():
    p = argparse.ArgumentParser(description="Approach 2 generative: BO, LLM regressor")
    p.add_argument("--property", default="esol", choices=list(PROPERTIES))
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--budget", type=int, default=15)
    p.add_argument("--batch", type=int, default=5, help="molecules picked per round")
    p.add_argument("--candidates", type=int, default=48, help="molecules generated/round")
    p.add_argument("--acq", default="ei", choices=["ei", "ucb"])
    p.add_argument("--kappa", type=float, default=2.0)
    p.add_argument("--xi", type=float, default=0.0)
    p.add_argument("--llm", default="gemini")
    p.add_argument("--vae", default=VAE_PATH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--auto", action="store_true", help="oracle stands in for the LLM")
    p.add_argument("--noise", type=float, default=0.0, help="--auto surrogate noise std")
    args = p.parse_args()

    prop = PROPERTIES[args.property]
    rng = np.random.default_rng(args.seed)
    vae = SelfiesVAE.load(args.vae)

    context = get_seeds(args.property)
    Zc, kept = vae.encode([m["smiles"] for m in context])
    by_smiles = {m["smiles"]: m for m in context}
    L_Z = list(Zc)
    L_y = [by_smiles[s]["score"] for s in kept]
    true_known = list(L_y)
    context = [by_smiles[s] for s in kept]
    evaluated = {m["smiles"] for m in context}
    f_best = max(m["score"] for m in context)
    best = max(context, key=lambda m: m["score"])

    results_dir = RESULTS_ROOT if args.property == "esol" else os.path.join(RESULTS_ROOT, args.property)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"approach2gen_{args.acq}_run{args.run}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    mode = "AUTO (oracle surrogate)" if args.auto else f"MANUAL LLM ({args.llm})"
    print(f"Approach 2 GEN ({args.property}) | Acq {args.acq.upper()} | Run {args.run} | "
          f"Budget {args.budget} | batch {args.batch} | {mode}")
    print(f"Logging to: {csv_path}")
    print(f"Starting best (seed): {best['name']} {prop['label']} = {best['score']:.3f}")

    eval_idx = 0
    rnd = 0
    while eval_idx < args.budget:
        rnd += 1
        gp = GP().fit(np.array(L_Z), np.array(L_y))

        top = np.argsort(np.array(true_known))[::-1][:min(6, len(L_Z))]
        anchor_Z = np.array([L_Z[i] for i in top])
        cand_smiles, cand_Z = generate_candidates(
            vae, anchor_Z, args.candidates, evaluated, rng)
        if not cand_smiles:
            print("VAE produced no new valid molecules; stopping early.")
            break

        mu, sigma = gp.posterior(cand_Z)
        order, scores = rank_candidates(mu, sigma, f_best, acq=args.acq,
                                        kappa=args.kappa, xi=args.xi)

        take = min(args.batch, args.budget - eval_idx, len(order))
        pick_pos = list(order[:take])
        pick_smiles = [cand_smiles[i] for i in pick_pos]

        raw_true = prop["score_batch"](pick_smiles)
        true_vals = [v if v is not None else 0.0 for v in raw_true]

        if args.auto:
            labels = label_picks_auto(pick_smiles, prop, rng, args.noise)
        else:
            labels = label_picks_manual(context, pick_smiles, prop)

        print("\n" + "=" * 60)
        print(f"=== Round {rnd} | picked {take} | evals {eval_idx}/{args.budget} ===")
        for pos, smi, tv, lab in zip(pick_pos, pick_smiles, true_vals, labels):
            eval_idx += 1
            tag = ""
            if tv > f_best:
                f_best, best = tv, {"name": "", "smiles": smi, "score": tv}
                tag = "  *** NEW BEST ***"
            gm, gs, av = float(mu[pos]), float(sigma[pos]), float(scores[pos])
            print(f"  #{eval_idx:2d} {smi:34s} GP mu={gm:.3f} sd={gs:.3f} "
                  f"acq={av:.3f} -> {prop['label']} {tv:.3f}{tag}")
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                    "eval_idx": eval_idx, "round": rnd, "run": args.run,
                    "llm": "auto" if args.auto else args.llm, "acq": args.acq,
                    "picked_smiles": smi, "gp_mu": f"{gm:.4f}", "gp_sigma": f"{gs:.4f}",
                    "acq_value": f"{av:.4f}",
                    "surrogate_label": "" if lab is None else f"{lab:.4f}",
                    "true_score": f"{tv:.4f}", "best_so_far": f"{f_best:.4f}",
                })

        for pos, smi, lab, tv in zip(pick_pos, pick_smiles, labels, true_vals):
            evaluated.add(smi)
            if lab is not None:
                L_Z.append(cand_Z[pos])
                L_y.append(lab)
                true_known.append(tv)
                context.append({"name": "", "smiles": smi, "score": lab})

    print("\n" + "=" * 60)
    print(f"DONE. Best: {best['smiles']}  {prop['label']} = {best['score']:.3f}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
