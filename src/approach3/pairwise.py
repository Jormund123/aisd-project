"""
Pairwise-ranking math for Approach 3 (Preferential BO with the LLM as a ranker).

The LLM never gives a number. It only answers duels: "which is more soluble, A or B?"
We collect many duels and fit a latent utility per molecule with a Bradley-Terry
model, then a preferential-BO acquisition function (UCB on the utility) picks the
next molecule to score with ESOL.

Everything here is pure (no I/O, no LLM, no files) so it can be unit-tested with
known-answer cases. A duel is a tuple (i, j, winner) where i, j are 0-based candidate
indices and winner is whichever of i or j the ranker judged more soluble.
"""

import numpy as np
from scipy.stats import spearmanr, kendalltau


def _split_winner_loser(duels):
    """Validate duels and return parallel winner/loser index arrays."""
    win, lose = [], []
    for d in duels:
        i, j, w = d
        if w != i and w != j:
            raise ValueError(f"duel winner {w} is neither {i} nor {j}")
        loser = j if w == i else i
        win.append(int(w))
        lose.append(int(loser))
    return np.asarray(win, dtype=int), np.asarray(lose, dtype=int)


def win_count_scores(duels, n):
    """Number of duels each of the n candidates won. Simple uncertainty-free rank."""
    scores = np.zeros(n, dtype=float)
    if not duels:
        return scores
    win, _ = _split_winner_loser(duels)
    for w in win:
        scores[w] += 1.0
    return scores


def bradley_terry(duels, n, reg=1e-3, max_iter=100, tol=1e-8):
    """Fit Bradley-Terry latent utilities by regularized MLE (Newton's method).

    Model: P(i beats j) = sigmoid(u_i - u_j). We maximize the log-likelihood of the
    observed duels minus an L2 ridge (0.5 * reg * ||u||^2). The ridge fixes the
    otherwise-unidentifiable global shift and keeps the Hessian invertible when duels
    are sparse. Utilities are mean-centered before returning.

    Returns (utility, std_err), both length-n arrays. std_err is sqrt of the diagonal
    of the inverse observed information (a rough per-candidate uncertainty that the
    UCB acquisition uses). With no duels, utility is all zeros and std_err is wide.
    """
    u = np.zeros(n, dtype=float)
    if not duels:
        return u, np.full(n, 1.0 / np.sqrt(reg))

    win, lose = _split_winner_loser(duels)

    for _ in range(max_iter):
        d = u[win] - u[lose]                 # utility gap, winner minus loser
        p = 1.0 / (1.0 + np.exp(-d))         # P(winner beats loser) under current u
        g = p * (1.0 - p)                    # logistic variance per duel

        grad = -reg * u
        np.add.at(grad, win, 1.0 - p)        # d log-lik / du_winner
        np.add.at(grad, lose, -(1.0 - p))    # d log-lik / du_loser

        # Hessian of the (concave) objective: negative definite thanks to the ridge.
        H = -reg * np.eye(n)
        np.add.at(H, (win, win), -g)
        np.add.at(H, (lose, lose), -g)
        np.add.at(H, (win, lose), g)
        np.add.at(H, (lose, win), g)

        step = np.linalg.solve(H, grad)      # Newton: u_new = u - H^-1 grad
        u = u - step
        if np.max(np.abs(step)) < tol:
            break

    u = u - u.mean()                         # center for identifiability

    # std_err from the inverse observed information (= inverse of -Hessian).
    d = u[win] - u[lose]
    p = 1.0 / (1.0 + np.exp(-d))
    g = p * (1.0 - p)
    info = reg * np.eye(n)
    np.add.at(info, (win, win), g)
    np.add.at(info, (lose, lose), g)
    np.add.at(info, (win, lose), -g)
    np.add.at(info, (lose, win), -g)
    cov = np.linalg.inv(info)
    std_err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return u, std_err


def pbo_acquisition(utility, std_err, evaluated_mask, kappa=2.0):
    """Preferential-BO acquisition: argmax over UNEVALUATED of utility + kappa*std_err.

    kappa=0 recovers greedy top-rank (pure exploit). Larger kappa rewards molecules
    the ranking is unsure about (explore). evaluated_mask is a boolean array; True
    means already ESOL-scored and therefore ineligible. Returns the chosen index, or
    None if every candidate is evaluated.
    """
    utility = np.asarray(utility, dtype=float)
    std_err = np.asarray(std_err, dtype=float)
    mask = np.asarray(evaluated_mask, dtype=bool)
    if mask.all():
        return None
    scores = utility + kappa * std_err
    scores = np.where(mask, -np.inf, scores)
    return int(np.argmax(scores))


def pairwise_accuracy(pred, true):
    """Fraction of candidate pairs ordered the same way by pred and by true.

    Pairs that are tied in either ranking are skipped. 1.0 = perfect ordering, 0.0 =
    perfectly reversed, ~0.5 = no relationship.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    n = len(pred)
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            sp = np.sign(pred[i] - pred[j])
            st = np.sign(true[i] - true[j])
            if sp == 0 or st == 0:
                continue
            total += 1
            if sp == st:
                correct += 1
    return correct / total if total else float("nan")


def ranking_metrics(pred_utility, true_logs):
    """How well the predicted ranking matches the true ESOL ranking.

    Returns {spearman, kendall, pairwise_accuracy}. This is the headline Approach-3
    measurement: it answers "is the LLM a good pairwise ranker?" against ground truth.
    """
    pred = np.asarray(pred_utility, dtype=float)
    true = np.asarray(true_logs, dtype=float)
    if len(pred) < 2:
        return {"spearman": float("nan"), "kendall": float("nan"),
                "pairwise_accuracy": float("nan")}
    rho = float(spearmanr(pred, true).statistic)
    tau = float(kendalltau(pred, true).statistic)
    return {"spearman": rho, "kendall": tau,
            "pairwise_accuracy": pairwise_accuracy(pred, true)}
