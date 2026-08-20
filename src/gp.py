"""
Minimal Gaussian Process regressor (numpy only) for the generative latent-space BO
in Approaches 2 and 3.

The GP is the uncertainty-aware surrogate the mentor's loop asks for: we fit it on
(latent z, label) pairs, where the label is whatever the LLM produced (a predicted
LogS for the regressor path, a Bradley-Terry utility for the ranker path). Its
posterior mean and standard deviation then feed the existing EI/UCB acquisition
functions in acquisition.py, evaluated at freshly decoded candidate latents.

Kept pure and hand-rolled (RBF kernel + Gaussian noise, lengthscale by marginal-
likelihood grid) so it is fully unit-testable, matches the repo's tested-math ethos
(see pairwise.py), and adds no heavy dependency. scikit-learn's GaussianProcessRegressor
is a drop-in fallback if ever needed.
"""

import numpy as np


def _rbf(A, B, lengthscale, signal_var):
    """RBF (squared-exponential) kernel between rows of A (n,d) and B (m,d)."""
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    d2 = np.clip(a2 + b2 - 2.0 * A @ B.T, 0.0, None)
    return signal_var * np.exp(-0.5 * d2 / (lengthscale ** 2))


class GP:
    """RBF-kernel GP with per-dimension input standardization and standardized targets.

    Usage:
        gp = GP().fit(Z, y)            # Z: (n, d), y: (n,)
        mu, sigma = gp.posterior(Zstar)  # both in the ORIGINAL y units
    """

    def __init__(self, noise=1e-2, signal_var=1.0,
                 lengthscales=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0)):
        self.noise = float(noise)
        self.signal_var = float(signal_var)
        self.lengthscale_grid = tuple(lengthscales)
        self._fitted = False

    # -- fitting -----------------------------------------------------------------
    def _log_marginal_likelihood(self, K, y):
        """LML of standardized y under kernel K (+ noise already added)."""
        n = len(y)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lml = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
        return lml, L, alpha

    def fit(self, Z, y):
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if Z.ndim == 1:
            Z = Z[:, None]
        n = len(y)

        # Standardize inputs per-dimension and targets (stabilizes the kernel).
        self.z_mean = Z.mean(axis=0)
        self.z_std = Z.std(axis=0)
        self.z_std[self.z_std < 1e-8] = 1.0
        Zs = (Z - self.z_mean) / self.z_std

        self.y_mean = float(y.mean())
        self.y_std = float(y.std())
        if self.y_std < 1e-8:
            self.y_std = 1.0
        ys = (y - self.y_mean) / self.y_std

        # Pick the lengthscale that maximizes the log marginal likelihood.
        best = None
        for ls in self.lengthscale_grid:
            K = _rbf(Zs, Zs, ls, self.signal_var) + self.noise * np.eye(n)
            lml, L, alpha = self._log_marginal_likelihood(K, ys)
            if best is None or lml > best[0]:
                best = (lml, ls, L, alpha)

        _, self.lengthscale, self._L, self._alpha = best
        self._Zs = Zs
        self._fitted = True
        return self

    # -- prediction --------------------------------------------------------------
    def posterior(self, Zstar):
        """Return (mu, sigma) at Zstar, both in original y units. sigma is a std dev."""
        if not self._fitted:
            raise RuntimeError("GP.posterior called before fit")
        Zstar = np.asarray(Zstar, dtype=float)
        if Zstar.ndim == 1:
            Zstar = Zstar[:, None]
        Zs = (Zstar - self.z_mean) / self.z_std

        Ks = _rbf(Zs, self._Zs, self.lengthscale, self.signal_var)  # (m, n)
        mu_s = Ks @ self._alpha                                     # standardized mean

        v = np.linalg.solve(self._L, Ks.T)                          # (n, m)
        var_s = self.signal_var - np.sum(v * v, axis=0)             # standardized var
        var_s = np.clip(var_s, 0.0, None)

        mu = mu_s * self.y_std + self.y_mean
        sigma = np.sqrt(var_s) * self.y_std
        return mu, sigma
