"""
Approach 3 on JNK3 (generative): Preferential BO with the LLM as a pairwise ranker,
searching the SELFIES-VAE latent space. Identical loop to optimizer_approach3_gen.py;
only the property changed (JNK3 score via TDC's Oracle instead of ESOL LogS).
Bradley-Terry, GP, and generate_candidates/rank_candidates are reused completely
unmodified.

Usage:
    python optimizer_approach3_gen_jnk3.py --run 1 --budget 15 --batch 5
    python optimizer_approach3_gen_jnk3.py --auto --budget 15            # validation
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "approach2"))

from jnk3_oracle import calculate_jnk3_batch           # noqa: E402
from vae import SelfiesVAE                             # noqa: E402
from gp import GP                                      # noqa: E402
from generative_bo import generate_candidates, rank_candidates  # noqa: E402
from pairwise import bradley_terry, ranking_metrics    # noqa: E402
from prompt_templates3_jnk3 import generate_pairwise_prompt  # noqa: E402
from optimizer_approach2_gen_jnk3 import build_seed_observed_jnk3  # noqa: E402
from optimizer_approach2 import read_pasted_json        # noqa: E402
from optimizer_approach3 import oracle_answers, parse_pairwise  # noqa: E402

VAE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "selfies_vae.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "approach3", "jnk3")

CSV_FIELDS = [
    "eval_idx", "round", "run", "llm", "picked_smiles",
    "gp_util_mu", "gp_sigma", "acq_value",
    "true_jnk3", "best_so_far", "n_duels_cum",
]


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


def answer_duels(pairs, context, mols, true_vals, auto, noise, rng):
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
    p = argparse.ArgumentParser(description="Approach 3 generative (JNK3): PBO, LLM ranker")
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--budget", type=int, default=15, help="total JNK3 oracle evaluations")
    p.add_argument("--batch", type=int, default=5, help="molecules picked per round")
    p.add_argument("--candidates", type=int, default=48, help="molecules generated/round")
    p.add_argument("--kappa", type=float, default=0.5, help="UCB exploration weight")
    p.add_argument("--llm", default="gemini")
    p.add_argument("--vae", default=VAE_PATH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--auto", action="store_true", help="JNK3 oracle answers the duels")
    p.add_argument("--noise", type=float, default=0.0, help="--auto duel flip probability")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    vae = SelfiesVAE.load(args.vae)

    seeds = build_seed_observed_jnk3()
    context = seeds
    mols = [{"name": m["name"], "smiles": m["smiles"]} for m in seeds]
    Zc, kept = vae.encode([m["smiles"] for m in mols])
    keep = set(kept)
    mols = [m for m in mols if m["smiles"] in keep]
    true_results = calculate_jnk3_batch([m["smiles"] for m in mols])
    true_vals = [r["jnk3_score"] if r is not None else 0.0 for r in true_results]
    Z_mols = list(Zc)
    duels = []
    evaluated = {m["smiles"] for m in mols}
    f_best = max(true_vals)
    best = {"smiles": mols[int(np.argmax(true_vals))]["smiles"], "jnk3": f_best}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"approach3gen_run{args.run}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    mode = "AUTO (JNK3 duels)" if args.auto else f"MANUAL LLM ({args.llm})"
    print(f"Approach 3 GEN (JNK3) | Run {args.run} | Budget {args.budget} | batch {args.batch} "
          f"| kappa {args.kappa} | {mode}")
    print(f"Logging to: {csv_path}")
    print(f"Starting best (seed): {best['smiles']} JNK3 = {best['jnk3']:.3f}")

    n_seed = len(mols)
    boot = [(i, i + 1) for i in range(n_seed - 1)]
    boot += [(0, i) for i in range(2, n_seed)]
    duels += answer_duels(boot, context, mols, true_vals, args.auto, args.noise, rng)
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
        pick_results = calculate_jnk3_batch(pick_smiles)
        true_pick = [r["jnk3_score"] if r is not None else 0.0 for r in pick_results]

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
        duels += answer_duels(pairs, context, mols, true_vals, args.auto, args.noise, rng)

        print("\n" + "=" * 60)
        print(f"=== Round {rnd} | picked {take} | evals {eval_idx}/{args.budget} "
              f"| duels {len(duels)} ===")
        for pos, smi, tv in zip(pick_pos, pick_smiles, true_pick):
            eval_idx += 1
            tag = ""
            if tv > f_best:
                f_best, best = tv, {"smiles": smi, "jnk3": tv}
                tag = "  *** NEW BEST ***"
            gm, gs, av = float(mu[pos]), float(sigma[pos]), float(scores[pos])
            print(f"  #{eval_idx:2d} {smi:34s} util={gm:.3f} sd={gs:.3f} "
                  f"acq={av:.3f} -> JNK3 {tv:.3f}{tag}")
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                    "eval_idx": eval_idx, "round": rnd, "run": args.run,
                    "llm": "auto" if args.auto else args.llm,
                    "picked_smiles": smi, "gp_util_mu": f"{gm:.4f}",
                    "gp_sigma": f"{gs:.4f}", "acq_value": f"{av:.4f}",
                    "true_jnk3": f"{tv:.4f}", "best_so_far": f"{f_best:.4f}",
                    "n_duels_cum": len(duels),
                })

    utility, _ = bradley_terry(duels, len(mols))
    rank_path = os.path.join(RESULTS_DIR, f"approach3gen_run{args.run}_ranking.csv")
    with open(rank_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "bt_utility", "true_jnk3"])
        for m, u, t in zip(mols, utility, true_vals):
            w.writerow([m["smiles"], f"{u:.4f}", f"{t:.4f}"])
    m = ranking_metrics(utility, np.array(true_vals))
    print("\n" + "=" * 60)
    print(f"DONE. Best: {best['smiles']}  JNK3 = {best['jnk3']:.3f}")
    print(f"Ranking quality (BT utility vs JNK3): Spearman {m['spearman']:+.3f} "
          f"| pairwise-acc {m['pairwise_accuracy']:.3f}")
    print(f"Results: {csv_path}\nRanking: {rank_path}")


if __name__ == "__main__":
    main()
