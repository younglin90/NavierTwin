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

    def _log_gauss(self, X: NDArray, mean: NDArray, cov: NDArray) -> NDArray:
        d = X.shape[1]
        diff = X - mean
        try:
            inv = np.linalg.inv(cov)
            det = float(np.linalg.det(cov))
        except np.linalg.LinAlgError:
            cov = cov + 1e-6 * np.eye(d)
            inv = np.linalg.inv(cov)
            det = float(np.linalg.det(cov))
        if det <= 0:
            det = 1e-30
        maha = np.einsum("ni,ij,nj->n", diff, inv, diff)
        return -0.5 * (d * np.log(2 * np.pi) + np.log(det) + maha)

    def fit(self, X: NDArray[np.float64], max_iter: int = 100, tol: float = 1e-6) -> "GMM":
        X = np.asarray(X, dtype=np.float64)
        self._init(X)
        ll_prev = -np.inf
        for _ in range(max_iter):
            # E
            logp = np.stack(
                [np.log(self.weights[k]) + self._log_gauss(X, self.means[k], self.covs[k])
                 for k in range(self.k)],
                axis=1,
            )
            # softmax
            m = logp.max(axis=1, keepdims=True)
            probs = np.exp(logp - m)
            probs = probs / probs.sum(axis=1, keepdims=True)

            # M
            Nk = probs.sum(axis=0)
            self.weights = Nk / X.shape[0]
            self.means = (probs.T @ X) / (Nk[:, None] + 1e-30)
            for k in range(self.k):
                diff = X - self.means[k]
                self.covs[k] = (
                    (probs[:, k:k + 1] * diff).T @ diff
                ) / (Nk[k] + 1e-30) + 1e-6 * np.eye(X.shape[1])

            ll = float(np.sum(m) + np.sum(np.log(np.exp(logp - m).sum(axis=1))))
            if abs(ll - ll_prev) < tol:
                break
            ll_prev = ll
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        X = np.asarray(X, dtype=np.float64)
        logp = np.stack(
            [np.log(self.weights[k]) + self._log_gauss(X, self.means[k], self.covs[k])
             for k in range(self.k)],
            axis=1,
        )
        return np.argmax(logp, axis=1).astype(np.int64)


__all__ = ["GMM"]
