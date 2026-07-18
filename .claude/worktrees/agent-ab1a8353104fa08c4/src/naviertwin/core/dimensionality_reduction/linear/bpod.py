"""Balanced POD (BPOD) — 관측/제어 가능성 기반 선형 차원축소.

    W_c = Σ (from direct simulation),   W_o = Σ (from adjoint simulation)
    (U, Σ, V) = SVD(W_o^{1/2} W_c^{1/2})   — balancing transformation.

실용 근사: 직접/수반 스냅샷 행렬 X (direct), Y (adjoint) 로부터
B = Yᵀ X 의 SVD 를 수행하여 balanced 모드 추출.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.bpod import BalancedPOD
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((40, 20))  # direct snapshots (n_features, n_snap)
    >>> Y = rng.standard_normal((40, 20))  # adjoint snapshots
    >>> bpod = BalancedPOD(n_modes=5)
    >>> bpod.fit(X, Y)
    >>> bpod.direct_modes_.shape
    (40, 5)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class BalancedPOD:
    """Method-of-snapshots 기반 BPOD."""

    def __init__(self, n_modes: int = 10) -> None:
        self.n_modes = n_modes
        self.direct_modes_: NDArray[np.float64] | None = None
        self.adjoint_modes_: NDArray[np.float64] | None = None
        self.singular_values_: NDArray[np.float64] | None = None
        self.is_fitted: bool = False

    def fit(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
    ) -> None:
        """direct X, adjoint Y 로부터 balanced 모드 학습.

        X, Y: (n_features, n_snap), 같은 n_snap.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.shape != Y.shape:
            raise ValueError(f"X, Y shape 불일치: {X.shape} vs {Y.shape}")

        # Hankel-like matrix H = Yᵀ X (n_snap × n_snap)
        H = Y.T @ X
        U, s, Vt = _svd(H, full_matrices=False)
        r = min(self.n_modes, s.size)

        # Balanced direct / adjoint modes
        # Phi = X V Σ^{-1/2}, Psi = Y U Σ^{-1/2}
        s_inv_sqrt = np.zeros(r)
        s_inv_sqrt[s[:r] > 1e-12] = 1.0 / np.sqrt(s[:r][s[:r] > 1e-12])
        self.direct_modes_ = X @ Vt[:r].T * s_inv_sqrt[None, :]
        self.adjoint_modes_ = Y @ U[:, :r] * s_inv_sqrt[None, :]
        self.singular_values_ = s[:r]
        self.is_fitted = True
        logger.info(
            "BalancedPOD 학습: r=%d, top σ=%.4g", r, float(s[0]) if s.size else 0.0,
        )

    def project(
        self, x: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """adjoint modes 로 직접 상태 x 를 balanced 좌표로 투영."""
        if not self.is_fitted or self.adjoint_modes_ is None:
            raise RuntimeError("fit() 먼저 호출")
        return self.adjoint_modes_.T @ np.asarray(x, dtype=np.float64)

    def reconstruct(
        self, coeffs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if not self.is_fitted or self.direct_modes_ is None:
            raise RuntimeError("fit() 먼저 호출")
        return self.direct_modes_ @ np.asarray(coeffs, dtype=np.float64)


__all__ = ["BalancedPOD"]
