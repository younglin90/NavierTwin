"""Gaussian Mixture Model — EM algorithm.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.gmm import GMM
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.standard_normal((100, 2)), rng.standard_normal((100, 2)) + 5])
    >>> gmm = GMM(k=2, seed=0).fit(X, max_iter=50)
    >>> gmm.means.shape
    (2, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


class GMM:
    def __init__(self, k: int = 2, seed: int | None = 0) -> None:
        self.k = int(k)
        self.seed = seed
        self.weights: NDArray | None = None
        self.means: NDArray | None = None
        self.covs: NDArray | None = None

    def _init(self, X: NDArray[np.float64]) -> None:
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        idx = rng.choice(n, size=self.k, replace=False)
        self.means = X[idx].copy()
        self.covs = np.stack([np.eye(d)] * self.k)
        self.weights = np.full(self.k, 1.0 / self.k)

    def _log_prob_matrix(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if _kernels is None:
            raise ImportError("naviertwin._native._kernels is required by GMM")
        return _kernels.gmm_log_prob_matrix(X, self.weights, self.means, self.covs)

    def fit(self, X: NDArray[np.float64], max_iter: int = 100, tol: float = 1e-6) -> "GMM":
        X = np.asarray(X, dtype=np.float64)
        self._init(X)
        ll_prev = -np.inf
        it = 0
        while it < max_iter:
            # E
            logp = self._log_prob_matrix(X)
            # softmax
            m = logp.max(axis=1, keepdims=True)
            probs = np.exp(logp - m)
            probs = probs / probs.sum(axis=1, keepdims=True)

            # M
            Nk = probs.sum(axis=0)
            self.weights = Nk / X.shape[0]
            self.means = (probs.T @ X) / (Nk[:, None] + 1e-30)
            k_idx = 0
            while k_idx < self.k:
                diff = X - self.means[k_idx]
                self.covs[k_idx] = (
                    (probs[:, k_idx:k_idx + 1] * diff).T @ diff
                ) / (Nk[k_idx] + 1e-30) + 1e-6 * np.eye(X.shape[1])
                k_idx += 1

            ll = float(np.sum(m) + np.sum(np.log(np.exp(logp - m).sum(axis=1))))
            if abs(ll - ll_prev) < tol:
                break
            ll_prev = ll
            it += 1
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        X = np.asarray(X, dtype=np.float64)
        logp = self._log_prob_matrix(X)
        return np.argmax(logp, axis=1).astype(np.int64)


__all__ = ["GMM"]
