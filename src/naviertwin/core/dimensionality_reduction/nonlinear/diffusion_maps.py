"""Diffusion Maps — 데이터 기하구조 보존 비선형 차원축소.

    K_ij = exp(-||x_i - x_j||² / (2σ²))
    D_ii = Σ_j K_ij
    P = D^{-1} K          (Markov transition)
    eigdecomp(P) → (1, λ_1, λ_2, ...) 의 eigenvectors 가 diffusion coords.

첫 번째 eigenvector 는 상수이므로 버리고 상위 k 개 사용.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.diffusion_maps import (
    ...     DiffusionMaps,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> # Swiss roll 스타일 저차원 구조
    >>> t = np.linspace(0, 4 * np.pi, 200)
    >>> X = np.column_stack([t * np.cos(t), t * np.sin(t), 0.1 * rng.standard_normal(200)])
    >>> dm = DiffusionMaps(n_components=2, epsilon=1.0)
    >>> Y = dm.fit_transform(X)
    >>> Y.shape
    (200, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class DiffusionMaps:
    """Diffusion maps 비선형 임베딩."""

    def __init__(
        self,
        n_components: int = 2,
        epsilon: float | None = None,
        alpha: float = 1.0,
    ) -> None:
        self.n_components = n_components
        self.epsilon = epsilon
        self.alpha = alpha

        self.eigenvalues_: NDArray[np.float64] | None = None
        self.eigenvectors_: NDArray[np.float64] | None = None
        self.epsilon_used_: float = 0.0

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        if N < 3:
            raise ValueError("샘플 수가 3 이상이어야 합니다")

        # Pairwise sq distance
        sqd = np.sum(X ** 2, axis=1, keepdims=True)
        D2 = sqd + sqd.T - 2.0 * X @ X.T
        D2 = np.maximum(D2, 0.0)

        # epsilon 자동 추정: 중앙 거리^2
        if self.epsilon is None:
            med = float(np.median(D2[D2 > 0])) if np.any(D2 > 0) else 1.0
            self.epsilon_used_ = med
        else:
            self.epsilon_used_ = float(self.epsilon)

        K = np.exp(-D2 / (2.0 * self.epsilon_used_))

        # α-normalization (Coifman-Lafon)
        q = K.sum(axis=1) ** self.alpha + 1e-30
        K = K / np.outer(q, q)

        d = K.sum(axis=1)
        P = K / d[:, None]

        # 비대칭 P 의 상위 고유쌍 — 대칭화 A = D^{-1/2} K D^{-1/2}
        d_sqrt = np.sqrt(d)
        A = K / np.outer(d_sqrt, d_sqrt)
        eigvals, eigvecs = np.linalg.eigh(A)
        # 내림차순
        idx = np.argsort(-eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 맵 = D^{-1/2} v
        psi = eigvecs / d_sqrt[:, None]

        self.eigenvalues_ = eigvals
        self.eigenvectors_ = psi
        logger.info(
            "DiffusionMaps fit: N=%d, ε=%.4g, top λ=%.4g",
            N, self.epsilon_used_, float(eigvals[0]),
        )
        # 첫 번째 eigenvector 는 상수 → 버리고 상위 n_components
        return psi[:, 1 : 1 + self.n_components] * eigvals[1 : 1 + self.n_components]


__all__ = ["DiffusionMaps"]
