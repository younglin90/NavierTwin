"""Moving Horizon Estimation — sliding-window Bayesian state estimate.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_assimilation.mhe import mhe_estimate
    >>> A = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> Y = np.array([[1.0], [2.0], [3.0]])
    >>> x0 = np.array([0.0, 0.0])
    >>> x = mhe_estimate(A, H, Y, x0, P0=np.eye(2), Q=0.01*np.eye(2), R=0.1*np.eye(1))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.data_assimilation.iterated_ekf import _right_solve


def mhe_estimate(
    A: NDArray[np.float64],
    H: NDArray[np.float64],
    Y: NDArray[np.float64],
    x0: NDArray[np.float64],
    *,
    P0: NDArray[np.float64],
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sliding-window Kalman filter (covariance form)."""
    N = Y.shape[0]
    x = x0.copy().astype(np.float64)
    P = P0.copy().astype(np.float64)
    k = 0
    while k < N:
        # predict
        x = A @ x
        P = A @ P @ A.T + Q
        # update
        S = H @ P @ H.T + R
        K = _right_solve(S, P @ H.T)
        x = x + K @ (Y[k] - H @ x)
        P = (np.eye(len(x)) - K @ H) @ P
        k += 1
    return x


__all__ = ["mhe_estimate"]
