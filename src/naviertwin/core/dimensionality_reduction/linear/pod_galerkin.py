"""POD-Galerkin 저차 동역학 — 선형 state-space.

스냅샷 X (n_feat, n_snap) → POD 모드 Φ (n_feat, r).
선형 연산자 A 를 X_{t+1} ≈ A X_t 에서 LS 로 추정 후 Â = Φᵀ A Φ.

    â_{t+1} = Â â_t + B̂ u_t
    x ≈ Φ â

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.pod_galerkin import (
    ...     PODGalerkinROM,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> # 합성 linear dynamics
    >>> A = np.diag([0.99, 0.97, 0.95])
    >>> X = np.zeros((3, 50))
    >>> X[:, 0] = [1, 1, 1]
    >>> k = 0
    >>> while k < 49:
    ...     X[:, k + 1] = A @ X[:, k]
    ...     k += 1
    >>> rom = PODGalerkinROM(n_modes=3)
    >>> rom.fit(X)
    >>> a0 = rom.encode(X[:, 0:1])
    >>> traj = rom.rollout(a0, n_steps=10)
    >>> traj.shape
    (10, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class PODGalerkinROM:
    """선형 POD-Galerkin ROM with optional input matrix."""

    def __init__(self, n_modes: int = 5) -> None:
        self.n_modes = n_modes
        self.modes_: NDArray[np.float64] | None = None
        self.mean_: NDArray[np.float64] | None = None
        self.A_hat_: NDArray[np.float64] | None = None
        self.B_hat_: NDArray[np.float64] | None = None
        self.is_fitted: bool = False

    def fit(
        self,
        snapshots: NDArray[np.float64],
        inputs: NDArray[np.float64] | None = None,
    ) -> None:
        """(n_feat, n_snap) 스냅샷 + optional (n_in, n_snap) 입력으로 학습.

        inputs 가 주어지면 Â·a + B̂·u LS 추정.
        """
        X = np.asarray(snapshots, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("snapshots (n_feat, n_snap) 2D 필요")

        self.mean_ = X.mean(axis=1)
        Xc = X - self.mean_[:, None]
        U, s, _ = _svd(Xc, full_matrices=False)
        r = min(self.n_modes, U.shape[1])
        self.modes_ = U[:, :r]

        # 잠재 좌표
        A_coef = self.modes_.T @ Xc  # (r, n_snap)
        # 선형 동역학 LS: A_coef[:, 1:] ≈ Â A_coef[:, :-1] + B̂ u_{:-1}
        A1 = A_coef[:, 1:]
        A0 = A_coef[:, :-1]
        if inputs is None:
            self.A_hat_ = A1 @ np.linalg.pinv(A0)
            self.B_hat_ = None
        else:
            U_in = np.asarray(inputs, dtype=np.float64)
            if U_in.ndim == 1:
                U_in = U_in[None, :]
            if U_in.shape[1] != A_coef.shape[1]:
                raise ValueError(
                    f"inputs 시간 길이({U_in.shape[1]}) != 스냅샷 길이({A_coef.shape[1]})"
                )
            U0 = U_in[:, :-1]
            Z = np.vstack([A0, U0])                 # (r + n_in, n_snap-1)
            M = A1 @ np.linalg.pinv(Z)               # (r, r + n_in)
            self.A_hat_ = M[:, :r]
            self.B_hat_ = M[:, r:]

        self.is_fitted = True
        logger.info(
            "POD-Galerkin fit: r=%d, input=%s",
            r, None if inputs is None else U_in.shape[0],
        )

    def encode(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted or self.modes_ is None or self.mean_ is None:
            raise RuntimeError("fit() 먼저 호출")
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.mean_[:, None]
        return self.modes_.T @ Xc

    def decode(self, a: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted or self.modes_ is None or self.mean_ is None:
            raise RuntimeError("fit() 먼저 호출")
        return self.modes_ @ np.asarray(a, dtype=np.float64) + self.mean_[:, None]

    def rollout(
        self,
        a0: NDArray[np.float64],
        n_steps: int,
        inputs: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """잠재 공간 roll-out. 반환 shape = (n_steps, r)."""
        if not self.is_fitted or self.A_hat_ is None:
            raise RuntimeError("fit() 먼저 호출")
        a = np.asarray(a0, dtype=np.float64).ravel()
        out = np.zeros((n_steps, a.size))
        k = 0
        while k < n_steps:
            if self.B_hat_ is not None and inputs is not None:
                u = np.asarray(inputs, dtype=np.float64)
                if u.ndim == 1:
                    u = u[None, :]
                a = self.A_hat_ @ a + self.B_hat_ @ u[:, k]
            else:
                a = self.A_hat_ @ a
            out[k] = a
            k += 1
        return out


__all__ = ["PODGalerkinROM"]
