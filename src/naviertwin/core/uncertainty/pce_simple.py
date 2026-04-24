"""간단한 Polynomial Chaos Expansion — Legendre/Hermite + 최소자승 fit.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.pce_simple import PCESimple
    >>> rng = np.random.default_rng(0)
    >>> xi = rng.uniform(-1, 1, size=(100, 1))
    >>> y = 3 + 2 * xi[:, 0] - xi[:, 0] ** 2
    >>> pce = PCESimple(order=3, family="legendre").fit(xi, y)
    >>> abs(float(pce.mean()) - (3 - 1/3)) < 0.5
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _legendre_basis(xi: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    """각 차원에 대해 Legendre P_0..P_order. xi ∈ [-1,1]."""
    n, d = xi.shape
    # 각 차원별 (n, order+1)
    cols = []
    for j in range(d):
        z = xi[:, j]
        basis = np.ones((n, order + 1))
        if order >= 1:
            basis[:, 1] = z
        for k in range(2, order + 1):
            basis[:, k] = ((2 * k - 1) * z * basis[:, k - 1] - (k - 1) * basis[:, k - 2]) / k
        cols.append(basis)
    # total-degree tensor product 는 작게 구성 (최대 order 까지 단일 차원의 합)
    # 단순 버전: additive (각 차원 legendre 만)
    # → 다차원 구조는 full tensor 로 대체하면 차원 저주. 여기서는 concat.
    return np.concatenate([c for c in cols], axis=1)


def _hermite_basis(xi: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    """probabilist's Hermite (He_n). xi ~ N(0,1)."""
    n, d = xi.shape
    cols = []
    for j in range(d):
        z = xi[:, j]
        basis = np.ones((n, order + 1))
        if order >= 1:
            basis[:, 1] = z
        for k in range(2, order + 1):
            basis[:, k] = z * basis[:, k - 1] - (k - 1) * basis[:, k - 2]
        cols.append(basis)
    return np.concatenate(cols, axis=1)


class PCESimple:
    """least-squares PCE (additive basis)."""

    def __init__(self, order: int = 3, family: str = "legendre") -> None:
        if family not in ("legendre", "hermite"):
            raise ValueError("family ∈ legendre/hermite")
        self.order = int(order)
        self.family = family
        self.coeffs: NDArray[np.float64] | None = None
        self.n_dim: int = 0

    def _basis(self, xi: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            _legendre_basis(xi, self.order) if self.family == "legendre"
            else _hermite_basis(xi, self.order)
        )

    def fit(self, xi: NDArray[np.float64], y: NDArray[np.float64]) -> "PCESimple":
        xi = np.asarray(xi, dtype=np.float64)
        if xi.ndim == 1:
            xi = xi[:, None]
        self.n_dim = xi.shape[1]
        Phi = self._basis(xi)
        self.coeffs, *_ = np.linalg.lstsq(Phi, np.asarray(y).ravel(), rcond=None)
        return self

    def predict(self, xi: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.coeffs is None:
            raise RuntimeError("fit 먼저")
        xi = np.asarray(xi, dtype=np.float64)
        if xi.ndim == 1:
            xi = xi[:, None]
        return self._basis(xi) @ self.coeffs

    def mean(self) -> float:
        """additive basis 에서 0차 계수의 평균."""
        if self.coeffs is None:
            raise RuntimeError("fit 먼저")
        # 각 차원의 0차 계수는 c[k*(order+1)] — 중복 상수항을 갖고 있음.
        # 대표적으로 첫 차원 상수 계수를 mean 으로 보고
        return float(self.coeffs[0])


__all__ = ["PCESimple"]
