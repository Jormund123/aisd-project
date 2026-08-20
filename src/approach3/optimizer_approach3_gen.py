"""
Approach 3 (generative): Preferential Bayesian Optimization with the LLM as a pairwise
ranker, searching the SELFIES-VAE latent space. Mentor's architecture, ranker variant.

The LLM never gives a number. It only judges duels ("which is more soluble, A or B?").
Per round (shared helpers in src/generative_bo.py):
  1. Bradley-Terry turns all duels so far into a latent utility per known molecule.
  2. A GP is fit on (latent z, utility) -> a smooth utility surface with uncertainty.
  3. NEW candidate molecules are generated from the VAE; the GP posterior (utility mean +
     uncertainty) feeds a UCB acquisition -> pick a small batch.
  4. ESOL (the oracle) scores the picks (budget) for the convergence curve; then the LLM
     duels each pick against strong/known references to place it in the ranking.
Round 1 has no duels yet, so utilities are flat and UCB picks the most novel molecules
(pure exploration); duels from those picks seed the ranking for later rounds.

Manual LLM stays copy-paste (duel prompt in, JSON A/B answers out). --auto answers duels
with the ESOL oracle (optionally noisy) so the whole loop is validated with no LLM.

Usage:
    python optimizer_approach3_gen.py --run 1 --budget 15 --batch 5
    python optimizer_approach3_gen.py --auto --budget 15            # validation
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "approach2"))

from esol import calculate_esol                       # noqa: E402
from vae import SelfiesVAE                             # noqa: E402
from gp import GP                                      # noqa: E402
from generative_bo import generate_candidates, rank_candidates  # noqa: E402
from pairwise import bradley_terry, ranking_metrics    # noqa: E402
from prompt_templates3 import generate_pairwise_prompt  # noqa: E402
from optimizer_approach2 import build_seed_observed, read_pasted_json  # noqa: E402
from optimizer_approach3 import oracle_answers, parse_pairwise         # noqa: E402

VAE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "selfies_vae.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "approach3")

CSV_FIELDS = [
    "eval_idx", "round", "run", "llm", "picked_smiles",
    "gp_util_mu", "gp_sigma", "acq_value",
    "true_logs_esol", "best_so_far", "n_duels_cum",
]


def make_duel_pairs(new_idxs, ref_idxs, n_mols, rng, n_rand=1):
    """Duels that place each new molecule: compare it against the given references (the
    soluble incumbents + current ranking leader) plus a random molecule, so a pick's
    standing propagates from real anchors. Returns (i, j) pairs into the molecule list."""
    pairs = []
    for idx in new_idxs:
        refs = [r for r in ref_idxs if r != idx]
        others = [r for r in range(n_mols) if r != idx and r not in refs]
        if others and n_rand:
            refs += list(rng.choice(others, size=min(n_rand, len(others)), replace=False))
        for r in dict.fromkeys(refs):            # dedupe, keep order
            pairs.append((idx, int(r)))
    return pairs


def answer_duels(pairs, context, mols, true_vals, auto, noise, rng):
    """Get duel outcomes from the ESOL oracle (--auto) or a pasted LLM response.

    context = seed molecules WITH true LogS (few-shot block); mols = the pool indexed by
    the duel pairs (no LogS shown, so the answer is not given away).
    """
    if auto:
        return oracle_answers(pairs, true_vals, noise, rng)
    print("\n----- COPY THIS PROMPT -----\n")
    print(generate_pairwise_prompt(context, mols, pairs))
    print("\n----- END PROMPT -----")
    while True:
        duels = parse_pairwise(read_pasted_json(), pairs)
        if duels:
            return duels
        print("No usable A/B answers parsed. Re-paste (does NOT use budget).")


def main():
    p = argparse.ArgumentParser(description="Approach 3 generative: PBO, LLM ranker")
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--budget", type=int, default=15, help="total ESOL oracle evaluations")
    p.add_argument("--batch", type=int, default=5, help="molecules picked per round")
    p.add_argument("--candidates", type=int, default=48, help="molecules generated/round")
    p.add_argument("--kappa", type=float, default=0.5, help="UCB exploration weight")
    p.add_argument("--llm", default="gemini")
    p.add_argument("--vae", default=VAE_PATH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--auto", action="store_true", help="ESOL oracle answers the duels")
    p.add_argument("--noise", type=float, default=0.0, help="--auto duel flip probability")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    vae = SelfiesVAE.load(args.vae)

    seeds = build_seed_observed()
    context = seeds  # seed molecules WITH true LogS, shown as the duel few-shot block
    mols = [{"name": m["name"], "smiles": m["smiles"]} for m in seeds]
    Zc, kept = vae.encode([m["smiles"] for m in mols])
    keep = set(kept)
    mols = [m for m in mols if m["smiles"] in keep]
    true_vals = [calculate_esol(m["smiles"])["logs_esol"] for m in mols]
    Z_mols = list(Zc)
    duels = []
    evaluated = {m["smiles"] for m in mols}
    f_best = max(true_vals)
    best = {"smiles": mols[int(np.argmax(true_vals))]["smiles"], "logs": f_best}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"approach3gen_run{args.run}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    mode = "AUTO (ESOL duels)" if args.auto else f"MANUAL LLM ({args.llm})"
    print(f"Approach 3 GEN | Run {args.run} | Budget {args.budget} | batch {args.batch} "
          f"| kappa {args.kappa} | {mode}")
    print(f"Logging to: {csv_path}")
    print(f"Starting best (seed): {best['smiles']} LogS = {best['logs']:+.3f}")

    # Bootstrap: rank the seed molecules with a batch of duels so round 1 already has a
    # utility gradient (seeds span the whole LogS range). Without this the first round has
    # no ranking signal and wastes budget on pure exploration.
    n_seed = len(mols)
    boot = [(i, i + 1) for i in range(n_seed - 1)]           # chain (connectivity)
    boot += [(0, i) for i in range(2, n_seed)]               # star (relative to seed 0)
    duels += answer_duels(boot, context, mols, true_vals, args.auto, args.noise, rng)
    print(f"Bootstrapped ranking with {len(boot)} seed duels.")

    eval_idx = 0
    rnd = 0
    while eval_idx < args.budget:
        rnd += 1
        utility, _ = bradley_terry(duels, len(mols))
        gp = GP().fit(np.array(Z_mols), utility)

        # Anchor generation on the incumbent region: the most soluble molecules actually
        # measured (true ESOL). Standard BO exploitation; the LLM ranker still decides
        # which generated candidate to pick via the GP-on-utility acquisition.
        top = np.argsort(np.array(true_vals))[::-1][:min(6, len(Z_mols))]
        anchor_Z = np.array([Z_mols[i] for i in top])
        cand_smiles, cand_Z = generate_candidates(
            vae, anchor_Z, args.candidates, evaluated, rng)
        if not cand_smiles:
            print("VAE produced no new valid molecules; stopping early.")
            break

        mu, sigma = gp.posterior(cand_Z)
        order, scores = rank_candidates(mu, sigma, f_best, acq="ucb", kappa=args.kappa)
        take = min(args.batch, args.budget - eval_idx, len(order))
        pick_pos = list(order[:take])
        pick_smiles = [cand_smiles[i] for i in pick_pos]
        true_pick = [calculate_esol(s)["logs_esol"] for s in pick_smiles]

        # Add picks to the molecule set (so duels can reference them).
        new_idxs = []
        for pos, smi, tv in zip(pick_pos, pick_smiles, true_pick):
            mols.append({"name": "", "smiles": smi})
            Z_mols.append(cand_Z[pos])
            true_vals.append(tv)
            evaluated.add(smi)
            new_idxs.append(len(mols) - 1)

        # References: the most soluble molecules measured so far (incumbents) plus the
        # current ranking leader, so each new pick is compared against strong anchors.
        by_true = list(np.argsort(np.array(true_vals))[::-1][:3])
        by_util = ([int(np.argmax(utility))] if len(utility) and
                   utility.max() > utility.min() else [])
        ref_idxs = list(dict.fromkeys(by_true + by_util))
        pairs = make_duel_pairs(new_idxs, ref_idxs, len(mols), rng)
        duels += answer_duels(pairs, context, mols, true_vals, args.auto, args.noise, rng)

        print("\n" + "=" * 60)
        print(f"=== Round {rnd} | picked {take} | evals {eval_idx}/{args.budget} "
              f"| duels {len(duels)} ===")
        for pos, smi, tv in zip(pick_pos, pick_smiles, true_pick):
            eval_idx += 1
            tag = ""
            if tv > f_best:
                f_best, best = tv, {"smiles": smi, "logs": tv}
                tag = "  *** NEW BEST ***"
            gm, gs, av = float(mu[pos]), float(sigma[pos]), float(scores[pos])
            print(f"  #{eval_idx:2d} {smi:34s} util={gm:+.2f} sd={gs:.2f} "
                  f"acq={av:.3f} -> ESOL {tv:+.3f}{tag}")
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                    "eval_idx": eval_idx, "round": rnd, "run": args.run,
                    "llm": "auto" if args.auto else args.llm,
                    "picked_smiles": smi, "gp_util_mu": f"{gm:.4f}",
                    "gp_sigma": f"{gs:.4f}", "acq_value": f"{av:.4f}",
                    "true_logs_esol": f"{tv:.4f}", "best_so_far": f"{f_best:.4f}",
                    "n_duels_cum": len(duels),
                })

    # Ranking-quality dump: final BT utility vs true ESOL for every molecule seen.
    utility, _ = bradley_terry(duels, len(mols))
    rank_path = os.path.join(RESULTS_DIR, f"approach3gen_run{args.run}_ranking.csv")
    with open(rank_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "bt_utility", "true_logs_esol"])
        for m, u, t in zip(mols, utility, true_vals):
            w.writerow([m["smiles"], f"{u:.4f}", f"{t:.4f}"])
    m = ranking_metrics(utility, np.array(true_vals))
    print("\n" + "=" * 60)
    print(f"DONE. Best: {best['smiles']}  LogS = {best['logs']:+.3f}")
    print(f"Ranking quality (BT utility vs ESOL): Spearman {m['spearman']:+.3f} "
          f"| pairwise-acc {m['pairwise_accuracy']:.3f}")
    print(f"Results: {csv_path}\nRanking: {rank_path}")


if __name__ == "__main__":
    main()
