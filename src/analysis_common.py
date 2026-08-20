"""
Shared analysis helpers for the generative (latent-space BO) runs of Approaches 2 and 3.

Kept dependency-light (numpy only) and self-contained so the two per-approach analysis
scripts, and the head-to-head comparison, all measure things the exact same way. The two
metrics the mentor asked for live here:

  1. top_k_so_far_by_round: mean oracle LogS of the k best molecules found so far, as a
     function of round. This is the optimization curve (how fast the search climbs).
  2. pairwise_accuracy: reduce ANY surrogate score (a regressor's predicted LogS OR a
     ranker's Bradley-Terry utility) to all pairwise orderings and score them against the
     ESOL oracle's ordering. This is the fair, apples-to-apples surrogate quality metric,
     so a regressor and a ranker can finally be compared on one axis.
"""

import numpy as np

SEED_BEST = 0.208          # methanol, the best seed molecule (the bar to beat)
RANDOM_BASELINE = 0.408    # results/baseline_summary.csv random_mean (15 draws x 1000)


def pairwise_accuracy(pred, true):
    """Fraction of molecule pairs the surrogate orders the same way ESOL does.

    pred = surrogate scores (regressor LogS or ranker utility), true = ESOL LogS.
    Ties in the oracle are skipped. 0.5 = coin flip, 1.0 = perfect ordering.
    """
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    good = total = 0
    n = len(pred)
    for i in range(n):
        for j in range(i + 1, n):
            if true[i] == true[j]:
                continue
            total += 1
            if np.sign(pred[i] - pred[j]) == np.sign(true[i] - true[j]):
                good += 1
    return good / total if total else float("nan")


def top_k_so_far_by_round(rounds, true_vals, k=3):
    """Mean oracle LogS of the k best molecules found up to and including each round.

    rounds = per-evaluation round index, true_vals = per-evaluation ESOL LogS (same
    order). Returns (round_numbers, mean_topk) so runs can be overlaid on one plot.
    """
    rounds = np.asarray(rounds, int)
    true_vals = np.asarray(true_vals, float)
    out_rounds, out_vals = [], []
    for r in sorted(set(rounds.tolist())):
        seen = true_vals[rounds <= r]
        topk = np.sort(seen)[::-1][:k]
        out_rounds.append(r)
        out_vals.append(float(np.mean(topk)))
    return out_rounds, out_vals


def regression_error(pred, true):
    """MAE, RMSE, bias (mean pred - true) for a regressor's predictions."""
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    err = pred - true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
    }
