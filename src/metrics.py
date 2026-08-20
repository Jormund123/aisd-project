"""
Shared evaluation metrics for all three approaches.

Pure functions, no I/O, no LLM, fully unit-testable. Two families:

1. Optimization metrics (apply identically to Approach 1/2/3 so the runs are
   comparable): simple regret, normalized score, sample efficiency (iterations to
   first improvement / to ceiling), success rate.
2. Quality metrics specific to a prediction style:
   - regression_metrics + sign_accuracy   -> Approach 2 (LLM predicts a number)
   - top1_hit + ndcg_at_k                  -> Approach 3 (LLM ranks)
     (Spearman / Kendall / pairwise accuracy live in pairwise.ranking_metrics.)
   - hit_rate                              -> Approach 1 (LLM proposes; no prediction)

"best_so_far" everywhere is the per-iteration best-so-far ESOL LogS trajectory (the
column logged by every optimizer). "start" is the seed best (iteration-0 value).
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Optimization metrics (shared across approaches)
# --------------------------------------------------------------------------- #

def best_found(best_so_far, start):
    """Best LogS achieved over the whole run, including the seed start."""
    traj = np.asarray(best_so_far, dtype=float)
    return float(max(traj.max(), start)) if traj.size else float(start)


def simple_regret(best_found_value, ceiling):
    """Gap between the best reachable value and what was actually found.

    Lower is better; 0 means the ceiling was reached. None ceiling (e.g. the
    open-ended Approach 1, which has no fixed candidate pool) -> nan.
    """
    if ceiling is None:
        return float("nan")
    return float(ceiling - best_found_value)


def normalized_score(best_found_value, start, ceiling):
    """Progress on a 0..1 scale: 0 = seed best, 1 = pool ceiling.

    Undefined when there is no headroom (ceiling <= start) or no ceiling -> nan.
    Can exceed 1 only if best_found beats the stated ceiling (shouldn't on a pool).
    """
    if ceiling is None or ceiling - start <= 1e-9:
        return float("nan")
    return float((best_found_value - start) / (ceiling - start))


def improvement_over_seed(best_found_value, start):
    """How much the run improved on the seed best (absolute LogS units)."""
    return float(best_found_value - start)


def first_improvement_iter(best_so_far, start, iterations=None):
    """Iteration at which best-so-far first strictly beat the seed best.

    Returns the matching value from `iterations` (1-based positions if not given),
    or None if the run never improved on the seed.
    """
    traj = np.asarray(best_so_far, dtype=float)
    iters = list(range(1, len(traj) + 1)) if iterations is None else list(iterations)
    for it, v in zip(iters, traj):
        if v > start + 1e-12:
            return int(it)
    return None


def iters_to_ceiling(best_so_far, ceiling, iterations=None, tol=1e-6):
    """Iteration at which best-so-far first reached the ceiling (within tol).

    Returns the matching `iterations` value, or None if the ceiling was never
    reached (or there is no ceiling).
    """
    if ceiling is None:
        return None
    traj = np.asarray(best_so_far, dtype=float)
    iters = list(range(1, len(traj) + 1)) if iterations is None else list(iterations)
    for it, v in zip(iters, traj):
        if v >= ceiling - tol:
            return int(it)
    return None


def success_rate(best_so_far, start):
    """Fraction of evaluations that strictly improved the running best.

    Each eval is compared to the best known just before it (the seed best for the
    first eval). 0 = no eval ever helped; 1 = every eval set a new best.
    """
    traj = np.asarray(best_so_far, dtype=float)
    if traj.size == 0:
        return float("nan")
    prev = np.concatenate(([start], traj[:-1]))
    return float(np.mean(traj > prev + 1e-12))


# --------------------------------------------------------------------------- #
# Approach 2: regression quality (LLM predicts a number)
# --------------------------------------------------------------------------- #

def regression_metrics(pred, true):
    """Accuracy of predicted LogS (mu) against true ESOL LogS.

    Returns mae, rmse, bias (mean pred-true), pearson, r2 (coefficient of
    determination of pred explaining true). Pearson is the same correlation
    analysis2 already printed as corr(mu, true); r2 here is pearson**2 only when
    pred is an unbiased estimator, so it is computed directly (1 - SS_res/SS_tot).
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    n = len(pred)
    if n == 0:
        return {k: float("nan") for k in ("mae", "rmse", "bias", "pearson", "r2")}
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    pearson = float(np.corrcoef(pred, true)[0, 1]) if n > 1 else float("nan")
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "bias": bias, "pearson": pearson, "r2": r2}


def sign_accuracy(pred, true, threshold):
    """Fraction of molecules placed on the correct side of `threshold`.

    Treats the task as "is this molecule more soluble than the threshold?" (e.g.
    the seed best). Measures whether the LLM at least gets the direction right even
    when the exact number is off. Ties on either side are skipped.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    sp = np.sign(pred - threshold)
    st = np.sign(true - threshold)
    keep = (sp != 0) & (st != 0)
    if not np.any(keep):
        return float("nan")
    return float(np.mean(sp[keep] == st[keep]))


# --------------------------------------------------------------------------- #
# Approach 3: ranking quality extras (LLM ranks; see pairwise.ranking_metrics too)
# --------------------------------------------------------------------------- #

def top1_hit(pred_utility, true):
    """1.0 if the top-ranked candidate by pred is also the true best, else 0.0.

    The single most decision-relevant ranking question: would trusting the LLM's
    #1 pick land on the actually-best molecule in the pool?
    """
    pred = np.asarray(pred_utility, dtype=float)
    true = np.asarray(true, dtype=float)
    if len(pred) == 0:
        return float("nan")
    return 1.0 if int(np.argmax(pred)) == int(np.argmax(true)) else 0.0


def ndcg_at_k(pred_utility, true, k=5):
    """Normalized Discounted Cumulative Gain over the top-k predicted candidates.

    Relevance = true LogS shifted so the worst candidate scores 0 (NDCG needs
    nonnegative gains). Rewards putting the genuinely most-soluble molecules near
    the top of the predicted ranking; 1.0 = ideal top-k ordering.
    """
    pred = np.asarray(pred_utility, dtype=float)
    true = np.asarray(true, dtype=float)
    n = len(pred)
    if n == 0:
        return float("nan")
    k = min(k, n)
    rel = true - true.min()                       # nonnegative gains
    if rel.max() <= 0:
        return float("nan")                       # all tied -> undefined
    pred_order = np.argsort(-pred, kind="stable")[:k]
    ideal_order = np.argsort(-rel, kind="stable")[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel[pred_order] * discounts))
    idcg = float(np.sum(rel[ideal_order] * discounts))
    return dcg / idcg if idcg > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Approach 1: proposal quality (LLM generates; no prediction to score)
# --------------------------------------------------------------------------- #

def hit_rate(values, start):
    """Fraction of proposals whose true ESOL LogS beat the seed best.

    Approach 1 makes no prediction, so its per-step quality is simply: how often
    does a freshly proposed molecule actually clear the bar we started from?
    """
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals > start + 1e-12))
