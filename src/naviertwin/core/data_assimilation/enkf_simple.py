"""경량 EnKF — 스토캐스틱 ensemble 갱신.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_assimilation.enkf_simple import EnKFSimple
    >>> rng = np.random.default_rng(0)
    >>> ens = rng.standard_normal((30, 2))  # 30 members, 2-dim state
    >>> kf = EnKFSimple(H=np.eye(2), R=np.eye(2)*0.1)
    >>> ens2 = kf.update(ens, z=np.array([1.0, 1.0]))
    >>> ens2.shape
    (30, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class EnKFSimple:
    """선형 관측 H + 대각 공분산 R 가정 stochastic EnKF."""

    def __init__(
        self, H: NDArray[np.float64], R: NDArray[np.float64],
    ) -> None:
        self.H = np.asarray(H, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)

    def update(
        self,
        ensemble: NDArray[np.float64],
        z: NDArray[np.float64],
        *, rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """ensemble (N, d) + 관측 z (m,) → posterior ensemble."""
        rng = rng or np.random.default_rng()
        X = np.asarray(ensemble, dtype=np.float64)
        N = X.shape[0]
        Xm = X.mean(axis=0, keepdims=True)
        A = X - Xm  # anomalies
        # sample covariance
        P = (A.T @ A) / (N - 1)
        # Kalman gain
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        # perturb observations
        m = self.R.shape[0]
        L = np.linalg.cholesky(self.R + 1e-12 * np.eye(m))
        perturb = rng.standard_normal((N, m)) @ L.T
        innovations = z[np.newaxis, :] + perturb - X @ self.H.T
        return X + innovations @ K.T


__all__ = ["EnKFSimple"]
