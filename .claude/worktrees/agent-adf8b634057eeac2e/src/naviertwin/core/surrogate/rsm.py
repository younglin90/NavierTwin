"""Response Surface Method — 다항 fit (full quadratic).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.surrogate.rsm import RSMQuadratic
    >>> X = np.random.default_rng(0).standard_normal((50, 2))
    >>> y = 3 + X[:, 0] - 2 * X[:, 1] + X[:, 0] ** 2 + 0.5 * X[:, 0] * X[:, 1]
    >>> rsm = RSMQuadratic().fit(X, y)
    >>> r2 = rsm.r_squared(X, y)
    >>> r2 > 0.99
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _quadratic_design(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """full quadratic design: constant, linear terms, and upper-triangle products."""
    n, d = X.shape
    tri_i, tri_j = np.triu_indices(d)
    quadratic = X[:, tri_i] * X[:, tri_j]
    return np.concatenate((np.ones((n, 1)), X, quadratic), axis=1)


class RSMQuadratic:
    def __init__(self) -> None:
        self.coeffs: NDArray | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "RSMQuadratic":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        D = _quadratic_design(X)
        self.coeffs, *_ = np.linalg.lstsq(D, y, rcond=None)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.coeffs is None:
            raise RuntimeError("fit 먼저")
        return _quadratic_design(np.asarray(X, dtype=np.float64)) @ self.coeffs

    def r_squared(
        self, X: NDArray[np.float64], y: NDArray[np.float64],
    ) -> float:
        y_hat = self.predict(X)
        y = np.asarray(y).ravel()
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-30
        return 1.0 - ss_res / ss_tot


__all__ = ["RSMQuadratic"]
