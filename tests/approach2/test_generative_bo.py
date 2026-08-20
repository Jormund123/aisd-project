"""
Unit tests for the shared generative-BO helpers (src/generative_bo.py).

The acquisition ranking is pure and tested directly. Candidate generation is tested
against a tiny stub VAE (no torch, no checkpoint) so the dedup / re-encode / short-batch
logic is checked deterministically. Run from the repo root:
    ./venv/bin/python -m pytest tests/approach2/test_generative_bo.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "approach2"))

from generative_bo import rank_candidates, generate_candidates  # noqa: E402


def test_rank_ucb_kappa_zero_is_greedy_on_mean():
    mu = np.array([0.0, 2.0, 1.0])
    sigma = np.array([5.0, 0.0, 0.0])
    order, _ = rank_candidates(mu, sigma, f_best=0.0, acq="ucb", kappa=0.0)
    assert order[0] == 1  # highest mean wins when uncertainty is ignored


def test_rank_ucb_large_kappa_rewards_uncertainty():
    mu = np.array([0.0, 2.0, 1.0])
    sigma = np.array([5.0, 0.0, 0.0])
    order, _ = rank_candidates(mu, sigma, f_best=0.0, acq="ucb", kappa=2.0)
    assert order[0] == 0  # huge sigma beats the higher mean


def test_rank_ei_prefers_higher_mean_when_sigma_equal():
    mu = np.array([0.0, 1.0, 2.0])
    sigma = np.array([1.0, 1.0, 1.0])
    order, scores = rank_candidates(mu, sigma, f_best=0.5, acq="ei")
    assert order[0] == 2
    assert np.all(scores >= 0.0)  # EI is non-negative


class StubVAE:
    """2-D latent toy VAE: molecule name encodes its integer latent, so decode/encode
    round-trip and different latents give different molecules."""
    latent_dim = 2

    def sample_latents(self, n, anchors=None, scale=1.0, jitter=0.5, rng=None):
        rng = rng or np.random.default_rng(0)
        # spread integer-ish latents so decode yields many distinct molecules
        return rng.integers(0, 50, size=(n, 2)).astype(float)

    def decode(self, Z):
        out = []
        for z in np.asarray(Z):
            a = int(round(z[0]))
            out.append(f"M{a}" if a % 7 else None)  # ~1 in 7 decodes invalid
        return out

    def encode(self, smiles_list):
        Z, kept = [], []
        for s in smiles_list:
            Z.append([float(s[1:]), 0.0])
            kept.append(s)
        return np.array(Z, dtype=float), kept


def test_generate_candidates_unique_and_excludes_seen():
    vae = StubVAE()
    rng = np.random.default_rng(1)
    exclude = {"M1", "M2"}
    smis, Z = generate_candidates(vae, anchor_Z=None, n_target=10,
                                  exclude_smiles=exclude, rng=rng)
    assert len(smis) == len(set(smis))          # all unique
    assert exclude.isdisjoint(smis)             # nothing already-seen
    assert None not in smis                      # invalids dropped
    assert Z.shape == (len(smis), 2)
    assert len(smis) <= 10


def test_generate_candidates_reencodes_consistently():
    vae = StubVAE()
    smis, Z = generate_candidates(vae, None, 5, set(), np.random.default_rng(2))
    # StubVAE.encode maps "M<k>" -> [k, 0]; confirm returned Z matches the molecule id
    for s, z in zip(smis, Z):
        assert z[0] == float(s[1:])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"ok   {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
