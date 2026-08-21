"""
Approach 3 (generative): Preferential Bayesian Optimization with the LLM as a pairwise
ranker, searching the SELFIES-VAE latent space. Works for any property registered in
common.PROPERTIES.

The LLM never gives a number. It only judges duels ("which is more X, A or B?"). Per
round: Bradley-Terry turns all duels so far into a latent utility per known molecule;
a GP is fit on (latent z, utility); NEW candidates come from the VAE; the GP posterior
feeds a UCB acquisition -> pick a small batch. The oracle scores the picks (budget)
for the convergence curve; the LLM duels each pick against strong/known references to
place it in the ranking.

Manual LLM stays copy-paste. --auto answers duels with the oracle (optionally noisy)
so the whole loop is validated with no LLM.

Usage:
    python _04_approach3.py --property esol --run 1 --budget 15 --batch 5
    python _04_approach3.py --property jnk3 --auto --budget 15            # validation
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    PROPERTIES, SelfiesVAE, GP, generate_candidates, rank_candidates,
    bradley_terry, ranking_metrics, oracle_answers, parse_pairwise,
    read_pasted_json,
)
from _01_seed_molecules import get_seeds  # noqa: E402

VAE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "selfies_vae.pt")
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "results", "approach3")

CSV_FIELDS = [
    "eval_idx", "round", "run", "llm", "picked_smiles",
    "gp_util_mu", "gp_sigma", "acq_value",
    "true_score", "best_so_far", "n_duels_cum",
]


def generate_pairwise_prompt(observed, candidates, pairs, prop):
    lines_obs = [f"{i:2d}. {m['smiles']}: {prop['label']} = {m['score']:.3f}"
                 for i, m in enumerate(observed, 1)]
    lines_duels = []
    for q, (i, j) in enumerate(pairs, 1):
        a, b = candidates[i], candidates[j]
        lines_duels.append(f"Q{q}: A = {a['smiles']}   vs   B = {b['smiles']}")
    return f"""You are a computational chemist comparing molecules by predicted {prop['prompt_property_desc']}.

{prop['domain_hint']}For reference, here are molecules already measured with their {prop['label']} values:

{chr(10).join(lines_obs)}

Below are pairs of molecules. For EACH pair, decide which molecule scores HIGHER on {prop['label']}. Do not estimate numbers, just choose A or B.

{chr(10).join(lines_duels)}

Respond ONLY with a JSON array in snippet code, one object per question, no other text:

[
  {{"q": 1, "winner": "A"}},
  {{"q": 2, "winner": "B"}}
]

Answer every question. "winner" must be exactly "A" or "B".
"""


def make_duel_pairs(new_idxs, ref_idxs, n_mols, rng, n_rand=1):
    pairs = []
    for idx in new_idxs:
        refs = [r for r in ref_idxs if r != idx]
        others = [r for r in range(n_mols) if r != idx and r not in refs]
        if others and n_rand:
            refs += list(rng.choice(others, size=min(n_rand, len(others)), replace=False))
        for r in dict.fromkeys(refs):
            pairs.append((idx, int(r)))
    return pairs


def answer_duels(pairs, context, mols, true_vals, prop, auto, noise, rng):
    if auto:
        return oracle_answers(pairs, true_vals, noise, rng)
    print("\n----- COPY THIS PROMPT -----\n")
    print(generate_pairwise_prompt(context, mols, pairs, prop))
    print("\n----- END PROMPT -----")
    while True:
        duels = parse_pairwise(read_pasted_json(), pairs)
        if duels:
            return duels
        print("No usable A/B answers parsed. Re-paste (does NOT use budget).")


def main():
    p = argparse.ArgumentParser(description="Approach 3 generative: PBO, LLM ranker")
    p.add_argument("--property", default="esol", choices=list(PROPERTIES))
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--budget", type=int, default=15)
    p.add_argument("--batch", type=int, default=5, help="molecules picked per round")
    p.add_argument("--candidates", type=int, default=48, help="molecules generated/round")
    p.add_argument("--kappa", type=float, default=0.5, help="UCB exploration weight")
    p.add_argument("--llm", default="gemini")
    p.add_argument("--vae", default=VAE_PATH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--auto", action="store_true", help="oracle answers the duels")
    p.add_argument("--noise", type=float, default=0.0, help="--auto duel flip probability")
    args = p.parse_args()

    prop = PROPERTIES[args.property]
    rng = np.random.default_rng(args.seed)
    vae = SelfiesVAE.load(args.vae)

    seeds = get_seeds(args.property)
    context = seeds
    mols = [{"name": m["name"], "smiles": m["smiles"]} for m in seeds]
    Zc, kept = vae.encode([m["smiles"] for m in mols])
    keep = set(kept)
    mols = [m for m in mols if m["smiles"] in keep]
    raw_true = prop["score_batch"]([m["smiles"] for m in mols])
    true_vals = [v if v is not None else 0.0 for v in raw_true]
    Z_mols = list(Zc)
    duels = []
    evaluated = {m["smiles"] for m in mols}
    f_best = max(true_vals)
    best = {"smiles": mols[int(np.argmax(true_vals))]["smiles"], "score": f_best}

    results_dir = RESULTS_ROOT if args.property == "esol" else os.path.join(RESULTS_ROOT, args.property)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"approach3gen_run{args.run}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    mode = "AUTO (oracle duels)" if args.auto else f"MANUAL LLM ({args.llm})"
    print(f"Approach 3 GEN ({args.property}) | Run {args.run} | Budget {args.budget} "
          f"| batch {args.batch} | kappa {args.kappa} | {mode}")
    print(f"Logging to: {csv_path}")
    print(f"Starting best (seed): {best['smiles']} {prop['label']} = {best['score']:.3f}")

    n_seed = len(mols)
    boot = [(i, i + 1) for i in range(n_seed - 1)]
    boot += [(0, i) for i in range(2, n_seed)]
    duels += answer_duels(boot, context, mols, true_vals, prop, args.auto, args.noise, rng)
    print(f"Bootstrapped ranking with {len(boot)} seed duels.")

    eval_idx = 0
    rnd = 0
    while eval_idx < args.budget:
        rnd += 1
        utility, _ = bradley_terry(duels, len(mols))
        gp = GP().fit(np.array(Z_mols), utility)

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
        raw_pick = prop["score_batch"](pick_smiles)
        true_pick = [v if v is not None else 0.0 for v in raw_pick]

        new_idxs = []
        for pos, smi, tv in zip(pick_pos, pick_smiles, true_pick):
            mols.append({"name": "", "smiles": smi})
            Z_mols.append(cand_Z[pos])
            true_vals.append(tv)
            evaluated.add(smi)
            new_idxs.append(len(mols) - 1)

        by_true = list(np.argsort(np.array(true_vals))[::-1][:3])
        by_util = ([int(np.argmax(utility))] if len(utility) and
                   utility.max() > utility.min() else [])
        ref_idxs = list(dict.fromkeys(by_true + by_util))
        pairs = make_duel_pairs(new_idxs, ref_idxs, len(mols), rng)
        duels += answer_duels(pairs, context, mols, true_vals, prop, args.auto, args.noise, rng)

        print("\n" + "=" * 60)
        print(f"=== Round {rnd} | picked {take} | evals {eval_idx}/{args.budget} "
              f"| duels {len(duels)} ===")
        for pos, smi, tv in zip(pick_pos, pick_smiles, true_pick):
            eval_idx += 1
            tag = ""
            if tv > f_best:
                f_best, best = tv, {"smiles": smi, "score": tv}
                tag = "  *** NEW BEST ***"
            gm, gs, av = float(mu[pos]), float(sigma[pos]), float(scores[pos])
            print(f"  #{eval_idx:2d} {smi:34s} util={gm:.3f} sd={gs:.3f} "
                  f"acq={av:.3f} -> {prop['label']} {tv:.3f}{tag}")
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                    "eval_idx": eval_idx, "round": rnd, "run": args.run,
                    "llm": "auto" if args.auto else args.llm,
                    "picked_smiles": smi, "gp_util_mu": f"{gm:.4f}",
                    "gp_sigma": f"{gs:.4f}", "acq_value": f"{av:.4f}",
                    "true_score": f"{tv:.4f}", "best_so_far": f"{f_best:.4f}",
                    "n_duels_cum": len(duels),
                })

    utility, _ = bradley_terry(duels, len(mols))
    rank_path = os.path.join(results_dir, f"approach3gen_run{args.run}_ranking.csv")
    with open(rank_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "bt_utility", "true_score"])
        for m, u, t in zip(mols, utility, true_vals):
            w.writerow([m["smiles"], f"{u:.4f}", f"{t:.4f}"])
    m = ranking_metrics(utility, np.array(true_vals))
    print("\n" + "=" * 60)
    print(f"DONE. Best: {best['smiles']}  {prop['label']} = {best['score']:.3f}")
    print(f"Ranking quality (BT utility vs {prop['label']}): Spearman {m['spearman']:+.3f} "
          f"| pairwise-acc {m['pairwise_accuracy']:.3f}")
    print(f"Results: {csv_path}\nRanking: {rank_path}")


if __name__ == "__main__":
    main()
