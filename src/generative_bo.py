"""
Shared engine for the generative latent-space Bayesian optimization used by
Approaches 2 (regressor) and 3 (ranker).

The two optimizers differ only in how molecules get a training label for the GP
surrogate (A2: the LLM's predicted LogS; A3: a Bradley-Terry utility from LLM duels).
Everything else, generating brand-new candidate molecules from the VAE latent space and
scoring them with an acquisition function, is identical and lives here as small, pure
helpers so both optimizers stay thin and this logic can be unit-tested.

Round shape (both approaches):
  1. label the known molecules (LLM or, in --auto, the ESOL oracle)
  2. GP.fit(latent Z of known molecules, labels)
  3. generate_candidates: sample latents near the known ones, decode to NEW molecules
  4. GP.posterior at the candidates -> acquisition score -> pick a small batch
  5. oracle-score the picks (counts against budget), record best-so-far, repeat
"""

import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "approach2"))

from acquisition import expected_improvement, upper_confidence_bound  # noqa: E402


def reasonable_molecule(smiles, min_heavy=3, max_heavy=40, max_mw=600.0):
    """Design-space sanity filter: keep drug-like sizes, reject decoder pathologies.

    The VAE occasionally decodes degenerate strings (e.g. C70 alkane chains -> ESOL -19)
    that would dominate and derail the search. We restrict candidates to a realistic size
    window, the generative analogue of Approach 1's structural constraints.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    heavy = mol.GetNumHeavyAtoms()
    if heavy < min_heavy or heavy > max_heavy:
        return False
    return Descriptors.MolWt(mol) <= max_mw


def generate_candidates(vae, anchor_Z, n_target, exclude_smiles, rng,
                        oversample=8, max_tries=10, scale=1.0, jitter=0.5,
                        accept_fn=reasonable_molecule):
    """Decode fresh, valid, unique molecules from the VAE latent space.

    Samples latents near the anchors (encoded known molecules) plus some from the
    prior, decodes them, drops invalids / duplicates / anything already seen, and
    RE-ENCODES the survivors so each returned molecule is paired with the latent that
    actually represents it (decode(z) then encode(smiles) is not identity).

    Returns (smiles_list, Z) with up to n_target molecules. May return fewer if the
    decoder is early in training; callers should handle a short batch.
    """
    seen = set(exclude_smiles)
    picked_smiles, picked_Z = [], []
    for _ in range(max_tries):
        if len(picked_smiles) >= n_target:
            break
        Z = vae.sample_latents(n_target * oversample, anchors=anchor_Z,
                               scale=scale, jitter=jitter, rng=rng)
        decoded = vae.decode(Z)
        fresh = []
        for smi in decoded:
            if smi and smi not in seen and (accept_fn is None or accept_fn(smi)):
                seen.add(smi)
                fresh.append(smi)
        if not fresh:
            continue
        Zf, kept = vae.encode(fresh)          # re-encode for molecule<->latent consistency
        for smi, z in zip(kept, Zf):
            picked_smiles.append(smi)
            picked_Z.append(z)
            if len(picked_smiles) >= n_target:
                break
    Z = np.array(picked_Z) if picked_Z else np.zeros((0, vae.latent_dim))
    return picked_smiles, Z


def rank_candidates(mu, sigma, f_best, acq="ei", kappa=2.0, xi=0.0):
    """Acquisition scores for candidates from GP posterior (mu, sigma).

    Reuses the Approach-2 acquisition functions. Returns (order, scores) where order
    indexes candidates best-first. We MAXIMIZE LogS (more soluble = better).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if acq == "ei":
        scores = expected_improvement(mu, sigma, f_best, xi=xi)
    elif acq == "ucb":
        scores = upper_confidence_bound(mu, sigma, kappa=kappa)
    else:
        raise ValueError(f"unknown acquisition '{acq}' (use 'ei' or 'ucb')")
    order = np.argsort(-scores)
    return order, scores
