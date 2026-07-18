"""Quadratic manifold ROM — x ≈ Φ z + (1/2) H (z ⊗ z).

Geelen, Wright, Willcox 2023 style; H 텐서를 least-squares 로 적합.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.quadratic_manifold import (
    ...     QuadraticManifold,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 30))
    >>> qm = QuadraticManifold(rank=3).fit(X)
    >>> qm.encode(X[:, :1]).shape
    (3, 1)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray


def _kron_z(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """unique pairs of z (z_i z_j, i<=j) — symmetric Kronecker."""
    rows, cols = np.triu_indices(z.shape[0])
    return np.asarray(z[rows] * z[cols])


def _kron_z_columns(Z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Column-wise symmetric Kronecker feature matrix."""
    rows, cols = np.triu_indices(Z.shape[0])
    return Z[rows, :] * Z[cols, :]


class QuadraticManifold:
    def __init__(self, rank: int = 5) -> None:
        self.rank = int(rank)
        self.Phi: NDArray | None = None
        self.H: NDArray | None = None  # (n, p) where p=r(r+1)/2

    def fit(self, X: NDArray[np.float64]) -> "QuadraticManifold":
        X = np.asarray(X, dtype=np.float64)
        # POD basis Φ
        U, _, _ = _svd(X, full_matrices=False)
        self.Phi = U[:, : self.rank]
        Z = self.Phi.T @ X  # (r, m)
        # residual after linear fit
        R = X - self.Phi @ Z
        # build feature matrix Z ⊗ Z (unique)
        Zk = _kron_z_columns(Z)  # (p, m)
        # least-squares: H Zk ≈ R
        H, *_ = np.linalg.lstsq(Zk.T, R.T, rcond=None)
        self.H = H.T  # (n, p)
        return self

    def encode(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.Phi is not None
        return self.Phi.T @ np.asarray(X, dtype=np.float64)

    def decode(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.Phi is not None and self.H is not None
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z[:, None]
        Zk = _kron_z_columns(Z)
        return self.Phi @ Z + self.H @ Zk


__all__ = ["QuadraticManifold"]
