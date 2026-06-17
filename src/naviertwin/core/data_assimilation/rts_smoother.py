"""RTS 스무더 — 선형 Kalman forward/backward 결합.

Examples:
    >>> import numpy as np
    >>> # 사용은 test 참조
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.data_assimilation.iterated_ekf import _right_solve


def rts_smoother(
    F: NDArray[np.float64], H: NDArray[np.float64],
    Q: NDArray[np.float64], R: NDArray[np.float64],
    x0: NDArray[np.float64], P0: NDArray[np.float64],
    measurements: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(means, covs) 스무딩. measurements (T, m)."""
    T = measurements.shape[0]
    d = x0.size
    # Forward (Kalman filter) results
    xf = np.zeros((T + 1, d))
    Pf = np.zeros((T + 1, d, d))
    xp = np.zeros((T + 1, d))
    Pp = np.zeros((T + 1, d, d))
    xf[0] = x0
    Pf[0] = P0
    xp[0] = x0
    Pp[0] = P0

    k = 0
    while k < T:
        # predict
        x_pred = F @ xf[k]
        P_pred = F @ Pf[k] @ F.T + Q
        xp[k + 1] = x_pred
        Pp[k + 1] = P_pred
        # update
        y = measurements[k] - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = _right_solve(S, P_pred @ H.T)
        xf[k + 1] = x_pred + K @ y
        Pf[k + 1] = (np.eye(d) - K @ H) @ P_pred
        k += 1

    # Backward
    xs = xf.copy()
    Ps = Pf.copy()
    k = T - 1
    while k >= 0:
        G = _right_solve(Pp[k + 1], Pf[k] @ F.T)
        xs[k] = xf[k] + G @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + G @ (Ps[k + 1] - Pp[k + 1]) @ G.T
        k -= 1
    return xs, Ps


__all__ = ["rts_smoother"]
