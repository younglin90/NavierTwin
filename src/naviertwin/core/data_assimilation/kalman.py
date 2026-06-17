"""선형 Kalman filter — 상태 x_k 를 prediction/update 로 추정.

x_{k+1} = F x_k + w,  w ~ N(0, Q)
z_k = H x_k + v,      v ~ N(0, R)

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_assimilation.kalman import KalmanFilter
    >>> kf = KalmanFilter(
    ...     F=np.eye(1), H=np.eye(1),
    ...     Q=np.eye(1)*1e-4, R=np.eye(1)*1e-2,
    ...     x0=np.zeros(1), P0=np.eye(1),
    ... )
    >>> _ = run_filter(kf, np.array([[1.0], [1.1], [0.9], [1.05]]))
    >>> abs(kf.x[0] - 1.0) < 0.2
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.data_assimilation.iterated_ekf import _right_solve


class KalmanFilter:
    """선형 가우시안 Kalman filter."""

    def __init__(
        self,
        F: NDArray[np.float64],
        H: NDArray[np.float64],
        Q: NDArray[np.float64],
        R: NDArray[np.float64],
        x0: NDArray[np.float64],
        P0: NDArray[np.float64],
    ) -> None:
        self.F = np.asarray(F, dtype=np.float64)
        self.H = np.asarray(H, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.x = np.asarray(x0, dtype=np.float64).copy()
        self.P = np.asarray(P0, dtype=np.float64).copy()

    def predict(self, u: NDArray | None = None, B: NDArray | None = None) -> None:
        """예측 단계."""
        self.x = self.F @ self.x
        if u is not None and B is not None:
            self.x = self.x + B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: NDArray[np.float64]) -> None:
        """측정 업데이트."""
        z = np.asarray(z, dtype=np.float64).ravel()
        y = z - self.H @ self.x           # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = _right_solve(S, self.P @ self.H.T)
        self.x = self.x + K @ y
        eye = np.eye(self.P.shape[0])
        self.P = (eye - K @ self.H) @ self.P


def run_filter(
    kf: KalmanFilter,
    measurements: NDArray[np.float64],
) -> NDArray[np.float64]:
    """z_k 시퀀스에 대해 predict/update 반복 → 상태 이력."""
    out = np.zeros((measurements.shape[0], kf.x.size), dtype=np.float64)
    i = 0
    while i < measurements.shape[0]:
        kf.predict()
        kf.update(measurements[i])
        out[i] = kf.x
        i += 1
    return out


__all__ = ["KalmanFilter", "run_filter"]
