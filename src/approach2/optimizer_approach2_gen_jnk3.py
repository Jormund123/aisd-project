"""
Approach 2 on JNK3 (generative): BO with the LLM as a regressor, searching the
SELFIES-VAE latent space. Identical loop to optimizer_approach2_gen.py; only the
property changed (JNK3 score via TDC's Oracle instead of ESOL LogS). VAE, GP,
acquisition functions, and generate_candidates/rank_candidates are reused completely
unmodified -- the GP standardizes internally, so no property-scale tuning is needed
(see docs/jnk3_implementation_plan.md).

Usage:
    python optimizer_approach2_gen_jnk3.py --run 1 --acq ei --budget 15 --batch 5
    python optimizer_approach2_gen_jnk3.py --auto --budget 15          # validation
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jnk3_oracle import calculate_jnk3_batch, validate_smiles  # noqa: E402
from seed_molecules_jnk3 import SEED_MOLECULES        # noqa: E402
from vae import SelfiesVAE                             # noqa: E402
from gp import GP                                      # noqa: E402
from generative_bo import generate_candidates, rank_candidates  # noqa: E402
from prompt_templates2_jnk3 import generate_regression_prompt   # noqa: E402
from optimizer_approach2 import read_pasted_json, extract_json_array  # noqa: E402

VAE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "selfies_vae.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "approach2", "jnk3")

CSV_FIELDS = [
    "eval_idx", "round", "run", "llm", "acq", "picked_smiles",
    "gp_mu", "gp_sigma", "acq_value", "surrogate_label",
    "true_jnk3", "best_so_far",
]


def build_seed_observed_jnk3():
    """Seed molecules with their JNK3 scores (see seed_molecules_jnk3.py)."""
    results = calculate_jnk3_batch([m["smiles"] for m in SEED_MOLECULES])
    observed = []
    for m, r in zip(SEED_MOLECULES, results):
        if r is None:
            continue
        observed.append({"name": m["name"], "smiles": r["canonical_smiles"],
                         "jnk3": r["jnk3_score"]})
    return observed


def match_predictions_jnk3(preds, candidates):
    """Same as optimizer_approach2.match_predictions, field renamed predicted_jnk3."""
    by_smiles = {c["smiles"]: j for j, c in enumerate(candidates)}
    idxs, mus, confs = [], [], []
    used = set()
    for p in preds:
        if not isinstance(p, dict) or "predicted_jnk3" not in p:
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
            mu = float(p["predicted_jnk3"])
            conf = float(p.get("confidence", 5))
        except (TypeError, ValueError):
            continue
        used.add(j)
        idxs.append(j)
        mus.append(mu)
        confs.append(conf)
    return idxs, mus, confs


def label_picks_auto(smiles_list, rng, noise):
    """--auto surrogate: the JNK3 oracle (optionally noisy) stands in for the LLM."""
    results = calculate_jnk3_batch(smiles_list)
    labels = []
    for r in results:
        y = r["jnk3_score"] if r is not None else 0.0
        if noise > 0:
            y += rng.normal(0.0, noise)
        labels.append(y)
    return labels


def label_picks_manual(context, smiles_list):
    candidates = [{"name": "", "smiles": s} for s in smiles_list]
    print("\n----- COPY THIS PROMPT -----\n")
    print(generate_regression_prompt(context, candidates))
    print("\n----- END PROMPT -----")
    preds = extract_json_array(read_pasted_json())
    labels = [None] * len(smiles_list)
    if not preds:
        return labels
    idxs, mus, _ = match_predictions_jnk3(preds, candidates)
    for j, mu in zip(idxs, mus):
        labels[j] = mu
    return labels


def main():
    p = argparse.ArgumentParser(description="Approach 2 generative (JNK3): BO, LLM regressor")
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--budget", type=int, default=15, help="total JNK3 oracle evaluations")
    p.add_argument("--batch", type=int, default=5, help="molecules picked per round")
    p.add_argument("--candidates", type=int, default=48, help="molecules generated/round")
    p.add_argument("--acq", default="ei", choices=["ei", "ucb"])
    p.add_argument("--kappa", type=float, default=2.0)
    p.add_argument("--xi", type=float, default=0.0)
    p.add_argument("--llm", default="gemini")
    p.add_argument("--vae", default=VAE_PATH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--auto", action="store_true", help="JNK3 oracle stands in for the LLM")
    p.add_argument("--noise", type=float, default=0.0, help="--auto surrogate noise std")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    vae = SelfiesVAE.load(args.vae)

    context = build_seed_observed_jnk3()
    Zc, kept = vae.encode([m["smiles"] for m in context])
    by_smiles = {m["smiles"]: m for m in context}
    L_Z = list(Zc)
    L_y = [by_smiles[s]["jnk3"] for s in kept]
    true_known = list(L_y)
    context = [by_smiles[s] for s in kept]
    evaluated = {m["smiles"] for m in context}
    f_best = max(m["jnk3"] for m in context)
    best = max(context, key=lambda m: m["jnk3"])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"approach2gen_{args.acq}_run{args.run}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    mode = "AUTO (JNK3 surrogate)" if args.auto else f"MANUAL LLM ({args.llm})"
    print(f"Approach 2 GEN (JNK3) | Acq {args.acq.upper()} | Run {args.run} | "
          f"Budget {args.budget} | batch {args.batch} | {mode}")
    print(f"Logging to: {csv_path}")
    print(f"Starting best (seed): {best['name']} JNK3 = {best['jnk3']:.3f}")

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

        true_results = calculate_jnk3_batch(pick_smiles)
        true_vals = [r["jnk3_score"] if r is not None else 0.0 for r in true_results]

        if args.auto:
            labels = label_picks_auto(pick_smiles, rng, args.noise)
        else:
            labels = label_picks_manual(context, pick_smiles)

        print("\n" + "=" * 60)
        print(f"=== Round {rnd} | picked {take} | evals {eval_idx}/{args.budget} ===")
        for pos, smi, tv, lab in zip(pick_pos, pick_smiles, true_vals, labels):
            eval_idx += 1
            tag = ""
            if tv > f_best:
                f_best, best = tv, {"name": "", "smiles": smi, "jnk3": tv}
                tag = "  *** NEW BEST ***"
            gm, gs, av = float(mu[pos]), float(sigma[pos]), float(scores[pos])
            print(f"  #{eval_idx:2d} {smi:34s} GP mu={gm:.3f} sd={gs:.3f} "
                  f"acq={av:.3f} -> JNK3 {tv:.3f}{tag}")
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                    "eval_idx": eval_idx, "round": rnd, "run": args.run,
                    "llm": "auto" if args.auto else args.llm, "acq": args.acq,
                    "picked_smiles": smi, "gp_mu": f"{gm:.4f}", "gp_sigma": f"{gs:.4f}",
                    "acq_value": f"{av:.4f}",
                    "surrogate_label": "" if lab is None else f"{lab:.4f}",
                    "true_jnk3": f"{tv:.4f}", "best_so_far": f"{f_best:.4f}",
                })

        for pos, smi, lab, tv in zip(pick_pos, pick_smiles, labels, true_vals):
            evaluated.add(smi)
            if lab is not None:
                L_Z.append(cand_Z[pos])
                L_y.append(lab)
                true_known.append(tv)
                context.append({"name": "", "smiles": smi, "jnk3": lab})

    print("\n" + "=" * 60)
    print(f"DONE. Best: {best['smiles']}  JNK3 = {best['jnk3']:.3f}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
