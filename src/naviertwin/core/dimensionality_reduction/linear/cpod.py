"""Constrained POD (cPOD) — 물리 보존법칙 제약을 만족하는 POD.

표준 POD 모드가 선형 제약 ``C u = d`` (예: 전역 질량 보존, 경계조건)
을 깨뜨릴 때, 각 모드를 제약 null-space 로 투영하여 보존성을 확보.

    U_c = (I - Cᵀ (C Cᵀ)^{-1} C) · U

이후 reconstruction 시 평균장만 제약을 만족하도록 보정.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.cpod import ConstrainedPOD
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((30, 20))  # (n_features, n_snapshots)
    >>> # 제약: 각 스냅샷의 합이 0 이어야 — Σ x_i = 0
    >>> C = np.ones((1, 30))
    >>> d = np.zeros(1)
    >>> cpod = ConstrainedPOD(n_modes=3, C=C, d=d)
    >>> cpod.fit(X)
    >>> X_rec = cpod.reconstruct(X)
    >>> float(abs((C @ X_rec - d[:, None]).max()))
    0.0
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import solve as _solve
from numpy.linalg import svd as _svd
from numpy.typing import NDArray

from naviertwin.core.dimensionality_reduction.base import BaseReducer
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class ConstrainedPOD(BaseReducer):
    """선형 제약을 만족하는 POD (모드 null-space 투영)."""

    def __init__(
        self,
        n_modes: int = 10,
        C: NDArray[np.float64] | None = None,
        d: NDArray[np.float64] | None = None,
    ) -> None:
        super().__init__()
        self.n_modes = n_modes
        C = np.asarray(C, dtype=np.float64) if C is not None else None
        d = np.asarray(d, dtype=np.float64) if d is not None else None
        if C is not None and d is None:
            raise ValueError("C 가 주어지면 d 도 필요합니다")
        if C is None and d is not None:
            raise ValueError("d 가 주어지면 C 도 필요합니다")
        if C is not None and d is not None:
            if C.ndim != 2 or C.shape[0] != d.size:
                raise ValueError(
                    f"C shape={C.shape}, d size={d.size} 불일치"
                )
        self.C = C
        self.d = d

        self.modes_: NDArray[np.float64] | None = None
        self.singular_values_: NDArray[np.float64] | None = None
        self.mean_: NDArray[np.float64] | None = None
        self.energy_ratio_: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------

    def _project_nullspace(self, M: NDArray[np.float64]) -> NDArray[np.float64]:
        """(n_features, k) 를 C 의 null-space 로 투영."""
        if self.C is None:
            return M
        C = self.C
        CCt = C @ C.T
        CM = C @ M  # (m, k)
        lam = _solve(CCt, CM)  # (m, k)
        return M - C.T @ lam

    def _constrained_mean(self, mean: NDArray[np.float64]) -> NDArray[np.float64]:
        """평균장 mean 을 C mean = d 로 보정."""
        if self.C is None:
            return mean
        C = self.C
        CCt = C @ C.T
        residual = C @ mean - self.d
        lam = _solve(CCt, residual)
        return mean - C.T @ lam

    def fit(self, snapshots: NDArray[np.float64]) -> None:
        """(n_features, n_snapshots) 로 학습."""
        X = np.asarray(snapshots, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"snapshots 2D 필요: {X.shape}")
        n_features, n_snap = X.shape
        n_modes = min(self.n_modes, n_snap, n_features)

        mean = X.mean(axis=1)
        mean = self._constrained_mean(mean)
        self.mean_ = mean
        X_c = X - mean[:, None]

        U, s, _ = _svd(X_c, full_matrices=False)
        U = U[:, :n_modes]
        # 제약 null-space 로 투영
        U_proj = self._project_nullspace(U)
        # orthonormalize (QR) — 투영 후 직교성 상실 복구
        Q, _ = np.linalg.qr(U_proj)
        self.modes_ = Q[:, :n_modes]
        self.singular_values_ = s[:n_modes]

        total = float(np.sum(s**2))
        cum = np.cumsum(s[:n_modes] ** 2) / max(total, 1e-30)
        self.energy_ratio_ = cum.astype(np.float64)

        self.n_components = n_modes
        self.is_fitted = True
        logger.info(
            "ConstrainedPOD 학습 완료: n_modes=%d, cum_energy=%.4f",
            n_modes,
            float(self.energy_ratio_[-1]),
        )

    def encode(self, snapshots: NDArray[np.float64]) -> NDArray[np.float64]:
        self._check_fitted()
        X = np.asarray(snapshots, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        X_c = X - self.mean_[:, None]
        return (self.modes_.T @ X_c).T

    def decode(self, coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
        self._check_fitted()
        c = np.asarray(coefficients, dtype=np.float64)
        if c.ndim == 1:
            c = c[None, :]
        rec = self.modes_ @ c.T + self.mean_[:, None]
        return rec

    def reconstruct(
        self,
        snapshots: NDArray[np.float64],
        n_modes: int | None = None,
    ) -> NDArray[np.float64]:
        self._check_fitted()
        if n_modes is None:
            n_modes = self.n_components
        if n_modes > self.n_components:
            raise ValueError(
                f"n_modes({n_modes}) > n_components({self.n_components})"
            )
        X = np.asarray(snapshots, dtype=np.float64)
        X_c = X - self.mean_[:, None]
        c = self.modes_[:, :n_modes].T @ X_c
        rec = self.modes_[:, :n_modes] @ c + self.mean_[:, None]
        return rec

    @property
    def energy_ratio(self) -> NDArray[np.float64]:
        self._check_fitted()
        return self.energy_ratio_


__all__ = ["ConstrainedPOD"]
