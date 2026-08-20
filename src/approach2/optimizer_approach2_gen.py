"""
Approach 2 (generative): Bayesian Optimization with the LLM as a regressor, searching
the SELFIES-VAE latent space instead of a fixed pool. This is the mentor's architecture.

Per round (see src/generative_bo.py for the shared helpers):
  1. Fit a GP on the latent vectors of the labeled molecules and their surrogate labels
     (LLM-predicted LogS; in --auto the ESOL oracle stands in for the LLM).
  2. Generate NEW candidate molecules by sampling latents near the known-good region and
     decoding them with the VAE.
  3. GP posterior (mean carries the LLM signal, std grows away from known data) feeds the
     existing EI/UCB acquisition -> pick a small batch.
  4. Score the picks with ESOL (the oracle; counts against budget) for the convergence
     curve, then ask the LLM to predict just those picks and add them to the labeled set.
The GP's uncertainty replaces Approach 2's uncalibrated 1-10 confidence; the oracle only
keeps score, it never trains the GP.

Manual LLM stays copy-paste: the script prints a prediction prompt for the picked
molecules and reads back a JSON array. --auto replaces that with the ESOL oracle so the
whole loop can be validated end to end with no LLM and no hallucination risk.

Usage:
    python optimizer_approach2_gen.py --run 1 --acq ei --budget 15 --batch 5
    python optimizer_approach2_gen.py --auto --budget 15          # validation
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from esol import calculate_esol                       # noqa: E402
from vae import SelfiesVAE                             # noqa: E402
from gp import GP                                      # noqa: E402
from generative_bo import generate_candidates, rank_candidates  # noqa: E402
from prompt_templates2 import generate_regression_prompt        # noqa: E402
from optimizer_approach2 import (                      # noqa: E402  (reuse manual I/O)
    build_seed_observed, read_pasted_json, extract_json_array, match_predictions,
)

VAE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "selfies_vae.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "approach2")

CSV_FIELDS = [
    "eval_idx", "round", "run", "llm", "acq", "picked_smiles",
    "gp_mu", "gp_sigma", "acq_value", "surrogate_label",
    "true_logs_esol", "best_so_far",
]


def label_picks_auto(smiles_list, rng, noise):
    """--auto surrogate: the ESOL oracle (optionally noisy) stands in for the LLM."""
    labels = []
    for smi in smiles_list:
        y = calculate_esol(smi)["logs_esol"]
        if noise > 0:
            y += rng.normal(0.0, noise)
        labels.append(y)
    return labels


def label_picks_manual(context, smiles_list):
    """Print the regression prompt for the picked molecules, read back predictions.

    Returns a label per molecule (None where the LLM gave no usable prediction).
    """
    candidates = [{"name": "", "smiles": s} for s in smiles_list]
    print("\n----- COPY THIS PROMPT -----\n")
    print(generate_regression_prompt(context, candidates))
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
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--budget", type=int, default=15, help="total ESOL oracle evaluations")
    p.add_argument("--batch", type=int, default=5, help="molecules picked per round")
    p.add_argument("--candidates", type=int, default=48, help="molecules generated/round")
    p.add_argument("--acq", default="ei", choices=["ei", "ucb"])
    p.add_argument("--kappa", type=float, default=2.0)
    p.add_argument("--xi", type=float, default=0.0)
    p.add_argument("--llm", default="gemini")
    p.add_argument("--vae", default=VAE_PATH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--auto", action="store_true", help="ESOL oracle stands in for the LLM")
    p.add_argument("--noise", type=float, default=0.0, help="--auto surrogate noise std")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    vae = SelfiesVAE.load(args.vae)

    # Known data: seeds with their true (given) ESOL values. The GP is bootstrapped on
    # these, then grows with LLM-labeled picks. context = few-shot block shown to the LLM.
    context = build_seed_observed()
    Zc, kept = vae.encode([m["smiles"] for m in context])
    by_smiles = {m["smiles"]: m for m in context}
    L_Z = list(Zc)
    L_y = [by_smiles[s]["logs"] for s in kept]   # surrogate labels (seed = true)
    true_known = list(L_y)                        # true ESOL of known molecules (incumbents)
    context = [by_smiles[s] for s in kept]
    evaluated = {m["smiles"] for m in context}
    f_best = max(m["logs"] for m in context)
    best = max(context, key=lambda m: m["logs"])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"approach2gen_{args.acq}_run{args.run}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    mode = "AUTO (ESOL surrogate)" if args.auto else f"MANUAL LLM ({args.llm})"
    print(f"Approach 2 GEN | Acq {args.acq.upper()} | Run {args.run} | Budget {args.budget} "
          f"| batch {args.batch} | {mode}")
    print(f"Logging to: {csv_path}")
    print(f"Starting best (seed): {best['name']} LogS = {best['logs']:+.3f}")

    eval_idx = 0
    rnd = 0
    while eval_idx < args.budget:
        rnd += 1
        gp = GP().fit(np.array(L_Z), np.array(L_y))

        # Anchor generation on the incumbent region: the most soluble molecules we have
        # actually measured (true ESOL). Standard BO exploitation; the surrogate still
        # decides which generated candidate to pick.
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

        # Oracle scores the picks (budget) for the convergence curve.
        true_vals = [calculate_esol(s)["logs_esol"] for s in pick_smiles]

        # Surrogate labels the picks -> added to the GP training set for next round.
        if args.auto:
            labels = label_picks_auto(pick_smiles, rng, args.noise)
        else:
            labels = label_picks_manual(context, pick_smiles)

        print("\n" + "=" * 60)
        print(f"=== Round {rnd} | picked {take} | evals {eval_idx}/{args.budget} ===")
        for k, (pos, smi, tv, lab) in enumerate(
                zip(pick_pos, pick_smiles, true_vals, labels)):
            eval_idx += 1
            tag = ""
            if tv > f_best:
                f_best, best = tv, {"name": "", "smiles": smi, "logs": tv}
                tag = "  *** NEW BEST ***"
            gm, gs, av = float(mu[pos]), float(sigma[pos]), float(scores[pos])
            print(f"  #{eval_idx:2d} {smi:34s} GP mu={gm:+.2f} sd={gs:.2f} "
                  f"acq={av:.3f} -> ESOL {tv:+.3f}{tag}")
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                    "eval_idx": eval_idx, "round": rnd, "run": args.run,
                    "llm": "auto" if args.auto else args.llm, "acq": args.acq,
                    "picked_smiles": smi, "gp_mu": f"{gm:.4f}", "gp_sigma": f"{gs:.4f}",
                    "acq_value": f"{av:.4f}",
                    "surrogate_label": "" if lab is None else f"{lab:.4f}",
                    "true_logs_esol": f"{tv:.4f}", "best_so_far": f"{f_best:.4f}",
                })

        # Grow the labeled set with successfully-labeled picks.
        for pos, smi, lab, tv in zip(pick_pos, pick_smiles, labels, true_vals):
            evaluated.add(smi)
            if lab is not None:
                L_Z.append(cand_Z[pos])
                L_y.append(lab)
                true_known.append(tv)
                context.append({"name": "", "smiles": smi, "logs": lab})

    print("\n" + "=" * 60)
    print(f"DONE. Best: {best['smiles']}  LogS = {best['logs']:+.3f}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
