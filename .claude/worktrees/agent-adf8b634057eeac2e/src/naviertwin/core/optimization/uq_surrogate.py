"""UQ surrogate — Polynomial Chaos Expansion (PCE).

    y(ξ) ≈ Σ_α c_α · Ψ_α(ξ)

ξ 는 균일 ([-1,1]^d) 분포 가정 → Legendre 다항식 기저.
tensor-product 기저 + 최소제곱 계수 피팅.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.uq_surrogate import PolynomialChaos
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-1, 1, (100, 2))
    >>> y = X[:, 0] ** 2 + 0.5 * X[:, 1]
    >>> pce = PolynomialChaos(degree=3)
    >>> pce.fit(X, y)
    >>> float(pce.mean_), float(pce.variance_)  # 통계 추정
    (...)
"""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _legendre(x: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """0..n 차 Legendre 다항식 값 배열."""
    P = np.ones((n + 1, x.size), dtype=np.float64)
    if n >= 1:
        P[1] = x
    k = 1
    while k < n:
        P[k + 1] = ((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1)
        k += 1
    return P


class PolynomialChaos:
    """PCE — Legendre tensor-product 기저 + least-squares."""

    def __init__(self, degree: int = 3) -> None:
        self.degree = degree
        self.multi_indices_: list[tuple[int, ...]] = []
        self.coef_: NDArray[np.float64] | None = None
        self.mean_: float = 0.0
        self.variance_: float = 0.0
        self.is_fitted: bool = False
        self.d_: int = 0

    def _make_indices(self, d: int) -> list[tuple[int, ...]]:
        """total-degree ≤ self.degree 인 multi-indices."""
        combos = product(range(self.degree + 1), repeat=d)
        return list(filter(lambda combo: sum(combo) <= self.degree, combos))

    def _design_matrix(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        d = X.shape[1]
        j_idx = np.arange(d)
        bases = np.stack(
            tuple(map(lambda j: _legendre(X[:, int(j)], self.degree), j_idx)),
            axis=0,
        )
        alphas = np.asarray(self.multi_indices_, dtype=int)
        term_values = np.empty((alphas.shape[0], d, X.shape[0]), dtype=np.float64)
        j = 0
        while j < d:
            term_values[:, j, :] = bases[j, alphas[:, j], :]
            j += 1
        return np.prod(term_values, axis=1).T

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2:
            raise ValueError("X (N, d) 2D 필요")
        self.d_ = X.shape[1]
        self.multi_indices_ = self._make_indices(self.d_)
        Psi = self._design_matrix(X)
        coef, *_ = np.linalg.lstsq(Psi, y, rcond=None)
        self.coef_ = coef

        # 평균 = constant term (α=0)
        zero_idx = self.multi_indices_.index(tuple([0] * self.d_))
        self.mean_ = float(coef[zero_idx])
        # 분산 = Σ_{α≠0} c_α² · <Ψ_α²>_ξ
        # Legendre 정규화: <P_n²> = 1/(2n+1) on [-1,1] (weight 1/2)
        alphas = np.asarray(self.multi_indices_, dtype=int)
        norms = np.prod(1.0 / (2 * alphas + 1), axis=1)
        active = np.sum(alphas, axis=1) > 0
        self.variance_ = float(np.sum(coef[active] ** 2 * norms[active]))

        self.is_fitted = True
        logger.info(
            "PCE 학습 완료: d=%d, terms=%d, mean=%.4g, var=%.4g",
            self.d_, len(self.multi_indices_), self.mean_, self.variance_,
        )

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted or self.coef_ is None:
            raise RuntimeError("fit() 먼저 호출")
        return self._design_matrix(X) @ self.coef_

    def sobol_indices(self) -> dict[str, NDArray[np.float64]]:
        """PCE 계수로부터 First-order / Total Sobol 지수 산출."""
        if not self.is_fitted or self.coef_ is None:
            raise RuntimeError("fit() 먼저 호출")
        d = self.d_
        total_var = max(self.variance_, 1e-30)

        # 각 축 j 에 대해 alpha_j > 0 이고 다른 축은 0 인 항이 first-order
        S1 = np.zeros(d)
        ST = np.zeros(d)
        alphas = np.asarray(self.multi_indices_, dtype=int)
        nonzero = alphas > 0
        active = np.sum(alphas, axis=1) > 0
        norms = np.prod(1.0 / (2 * alphas + 1), axis=1)
        contrib = self.coef_**2 * norms / total_var
        single_axis = np.sum(nonzero, axis=1) == 1
        np.add.at(S1, np.argmax(nonzero[single_axis], axis=1), contrib[single_axis])
        j = 0
        while j < d:
            ST[j] = float(np.sum(contrib[active & nonzero[:, j]]))
            j += 1
        return {"S1": S1, "ST": ST}


__all__ = ["PolynomialChaos"]
