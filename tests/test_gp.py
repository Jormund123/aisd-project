"""
Known-answer unit tests for src/gp.py (the latent-space BO surrogate).

No torch, no files, deterministic. Run from the repo root:
    ./venv/bin/python -m pytest tests/test_gp.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from common import GP, _rbf  # noqa: E402


def test_rbf_diagonal_is_signal_var():
    X = np.array([[0.0], [1.0], [2.0]])
    K = _rbf(X, X, lengthscale=1.0, signal_var=2.0)
    assert np.allclose(np.diag(K), 2.0)
    assert np.all(K <= 2.0 + 1e-9)  # off-diagonal never exceeds the diagonal


def test_rbf_decreases_with_distance():
    a = np.array([[0.0]])
    near = _rbf(a, np.array([[0.5]]), 1.0, 1.0)[0, 0]
    far = _rbf(a, np.array([[3.0]]), 1.0, 1.0)[0, 0]
    assert near > far


def test_gp_interpolates_training_points():
    # Low noise -> posterior mean should nearly reproduce the targets at train inputs.
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(12, 3))
    y = np.sin(Z[:, 0]) + 0.5 * Z[:, 1]
    gp = GP(noise=1e-6).fit(Z, y)
    mu, _ = gp.posterior(Z)
    assert np.max(np.abs(mu - y)) < 1e-2


def test_gp_uncertainty_grows_away_from_data():
    Z = np.linspace(-2, 2, 9)[:, None]
    y = np.sin(Z).ravel()
    gp = GP(noise=1e-4).fit(Z, y)
    _, sigma_near = gp.posterior(np.array([[0.0]]))   # inside the data
    _, sigma_far = gp.posterior(np.array([[50.0]]))   # far outside
    assert sigma_far[0] > sigma_near[0]


def test_gp_far_point_reverts_toward_prior_mean():
    Z = np.linspace(-1, 1, 7)[:, None]
    y = 3.0 + 0.0 * Z.ravel()          # constant target = 3.0
    gp = GP(noise=1e-4).fit(Z, y)
    mu_far, _ = gp.posterior(np.array([[100.0]]))
    # Far from data the mean returns to the target average (~3.0).
    assert abs(mu_far[0] - 3.0) < 1e-3


def test_gp_constant_target_is_stable():
    # y_std = 0 must not divide-by-zero; predictions stay at the constant.
    Z = np.random.default_rng(1).normal(size=(6, 2))
    y = np.full(6, -1.5)
    gp = GP().fit(Z, y)
    mu, sigma = gp.posterior(Z)
    assert np.allclose(mu, -1.5, atol=1e-6)
    assert np.all(sigma >= 0.0)


def test_gp_posterior_shapes():
    Z = np.random.default_rng(2).normal(size=(10, 4))
    y = Z.sum(axis=1)
    gp = GP().fit(Z, y)
    mu, sigma = gp.posterior(np.random.default_rng(3).normal(size=(5, 4)))
    assert mu.shape == (5,)
    assert sigma.shape == (5,)
    assert np.all(sigma >= 0.0)


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
