"""Bayesian linear regression — conjugate Gaussian posterior.

Prior: w ~ N(0, α⁻¹ I), likelihood: y = Φw + ε, ε ~ N(0, β⁻¹).
Posterior: w ~ N(μ, Σ), μ = β Σ Φᵀ y, Σ = (α I + β ΦᵀΦ)⁻¹.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.surrogate.bayesian_linear import BayesianLinearRegression
    >>> X = np.linspace(0, 1, 20)[:, None]
    >>> y = 2 * X[:, 0] + 1 + 0.1 * np.random.default_rng(0).standard_normal(20)
    >>> Phi = np.hstack([np.ones_like(X), X])
    >>> blr = BayesianLinearRegression(alpha=1.0, beta=100.0).fit(Phi, y)
    >>> mu, var = blr.predict(Phi)
    >>> var[0] > 0
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class BayesianLinearRegression:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.mean_: NDArray[np.float64] | None = None
        self.cov_: NDArray[np.float64] | None = None

    def fit(self, Phi: NDArray[np.float64], y: NDArray[np.float64]) -> "BayesianLinearRegression":
        Phi = np.asarray(Phi, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        d = Phi.shape[1]
        Sinv = self.alpha * np.eye(d) + self.beta * Phi.T @ Phi
        S = np.linalg.inv(Sinv)
        mu = self.beta * S @ Phi.T @ y
        self.mean_ = mu
        self.cov_ = S
        return self

    def predict(
        self, Phi: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        assert self.mean_ is not None and self.cov_ is not None
        Phi = np.asarray(Phi, dtype=np.float64)
        mu = Phi @ self.mean_
        # predictive variance = 1/β + φᵀ S φ
        var = 1.0 / self.beta + np.einsum("ij,jk,ik->i", Phi, self.cov_, Phi)
        return mu, var

    def sample_weights(
        self, n: int = 10, rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        assert self.mean_ is not None and self.cov_ is not None
        rng = rng or np.random.default_rng()
        return rng.multivariate_normal(self.mean_, self.cov_, size=n)


__all__ = ["BayesianLinearRegression"]
