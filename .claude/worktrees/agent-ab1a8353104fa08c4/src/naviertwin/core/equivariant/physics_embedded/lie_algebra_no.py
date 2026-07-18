"""Lie Algebra Canonicalization — SO(2) 회전 canonicalization 예.

2D 벡터 필드에서 각 샘플을 "표준 방향" 으로 회전한 뒤 base operator 에
입력. 예측 후 원래 방향으로 복원. SO(2) 에 대해 exact-equivariant.

References:
    Kaba et al., "Equivariance via canonicalization", ICLR 2023.
    Lie Algebra Canonicalization, ICLR 2025 (general).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.equivariant.physics_embedded.lie_algebra_no import (
    ...     SO2Canonicalizer,
    ... )
    >>> u = np.array([[1.0, 0.5], [-0.3, 0.2]])
    >>> can = SO2Canonicalizer()
    >>> u_canon, theta = can.canonicalize(u)
    >>> u_back = can.decanonicalize(u_canon, theta)
    >>> float(np.linalg.norm(u - u_back)) < 1e-10
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class SO2Canonicalizer:
    """벡터 필드 평균 방향을 +x 로 정렬하는 canonicalization.

    canonicalize(u) → (u', θ)  with u' = R(-θ) u, θ = angle of mean u.
    decanonicalize(u', θ) = R(θ) u'.
    """

    def canonicalize(
        self, u: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float]:
        u = np.asarray(u, dtype=np.float64)
        if u.ndim != 2 or u.shape[-1] != 2:
            raise ValueError(f"u (N, 2) 2D 필요: {u.shape}")
        mean_v = u.mean(axis=0)
        theta = float(np.arctan2(mean_v[1], mean_v[0]))
        c, s = np.cos(theta), np.sin(theta)
        R_neg = np.array([[c, s], [-s, c]])  # R(-θ)
        return u @ R_neg.T, theta

    def decanonicalize(
        self, u: NDArray[np.float64], theta: float
    ) -> NDArray[np.float64]:
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return u @ R.T


class SO2EquivariantOperator:
    """Canonicalization-based SO(2)-equivariant operator wrapper.

    base_fn: np.ndarray (N, 2) → np.ndarray (..., 2) — 사용자의 모델.
    """

    def __init__(
        self,
        base_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> None:
        self.base_fn = base_fn
        self._can = SO2Canonicalizer()

    def __call__(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        u_canon, theta = self._can.canonicalize(u)
        y_canon = self.base_fn(u_canon)
        y_canon = np.asarray(y_canon, dtype=np.float64)
        if y_canon.shape[-1] != 2:
            # 벡터 출력이 아니면 그대로 반환 (scalar equivariance 는 identity)
            return y_canon
        return self._can.decanonicalize(y_canon, theta)


__all__ = ["SO2Canonicalizer", "SO2EquivariantOperator"]
