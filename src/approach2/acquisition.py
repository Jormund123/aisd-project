"""
Acquisition functions for Approach 2 (BO with the LLM as a regressor).

The LLM predicts a mean LogS (mu) and a confidence (1-10) for each candidate.
We turn confidence into an uncertainty (sigma), then score each candidate with an
acquisition function and pick the argmax. We MAXIMIZE LogS (more soluble = better).

Two acquisitions:
  - Expected Improvement (EI): balances exploit (high mu) and explore (high sigma).
  - Upper Confidence Bound (UCB): mu + kappa * sigma.

Confidence -> sigma is the open research choice (the LLM's 1-10 is not calibrated).
We use a simple linear map: confidence 10 -> sigma_min (sure), 1 -> sigma_max (unsure).
"""

import numpy as np
from scipy.stats import norm


def confidence_to_sigma(confidence, sigma_min=0.1, sigma_max=2.0):
    """Map an LLM confidence in [1, 10] to a standard deviation.

    conf = 10 -> sigma_min (very sure, tight), conf = 1 -> sigma_max (unsure, wide).
    Linear in between. Values outside [1, 10] are clamped. Returns a float or array.
    """
    conf = np.clip(np.asarray(confidence, dtype=float), 1.0, 10.0)
    frac = (10.0 - conf) / 9.0  # 0 at conf=10, 1 at conf=1
    return sigma_min + (sigma_max - sigma_min) * frac


def expected_improvement(mu, sigma, f_best, xi=0.0):
    """EI for maximization. mu, sigma may be scalars or arrays.

    EI = (mu - f_best - xi) * Phi(z) + sigma * phi(z),  z = (mu - f_best - xi)/sigma
    xi is an exploration margin (larger xi favors exploration). Where sigma <= 0 the
    EI collapses to max(mu - f_best - xi, 0) (no uncertainty bonus).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    improvement = mu - f_best - xi
    ei = np.zeros_like(mu)

    pos = sigma > 0
    z = np.zeros_like(mu)
    z[pos] = improvement[pos] / sigma[pos]
    ei[pos] = improvement[pos] * norm.cdf(z[pos]) + sigma[pos] * norm.pdf(z[pos])
    ei[~pos] = np.maximum(improvement[~pos], 0.0)
    return ei


def upper_confidence_bound(mu, sigma, kappa=2.0):
    """UCB for maximization: mu + kappa * sigma. Larger kappa explores more."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return mu + kappa * sigma


def score_candidates(mu, confidence, f_best, acq="ei",
                     kappa=2.0, xi=0.0, sigma_min=0.1, sigma_max=2.0):
    """Full pipeline: confidence -> sigma -> acquisition scores.

    Returns (scores, sigma) as numpy arrays aligned with the inputs.
    """
    sigma = confidence_to_sigma(confidence, sigma_min, sigma_max)
    if acq == "ei":
        scores = expected_improvement(mu, sigma, f_best, xi=xi)
    elif acq == "ucb":
        scores = upper_confidence_bound(mu, sigma, kappa=kappa)
    else:
        raise ValueError(f"unknown acquisition '{acq}' (use 'ei' or 'ucb')")
    return np.asarray(scores, dtype=float), np.asarray(sigma, dtype=float)
