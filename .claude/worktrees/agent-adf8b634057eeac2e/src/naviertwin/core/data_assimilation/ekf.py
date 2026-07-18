"""Extended Kalman Filter — nonlinear f, h with linearization.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_assimilation.ekf import ekf_step
    >>> x, P = np.zeros(2), np.eye(2)
    >>> def f(x): return x
    >>> def F(x): return np.eye(2)
    >>> def h(x): return x[:1]
    >>> def H(x): return np.eye(2)[:1]
    >>> x2, P2 = ekf_step(x, P, f, F, h, H, z=np.array([1.0]),
    ...                    Q=0.01*np.eye(2), R=np.eye(1))
    >>> x2.shape
    (2,)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def ekf_step(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    f: Callable[[NDArray], NDArray],
    F: Callable[[NDArray], NDArray],
    h: Callable[[NDArray], NDArray],
    H: Callable[[NDArray], NDArray],
    *,
    z: NDArray[np.float64],
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One EKF predict + update."""
    # predict
    x_pred = f(x)
    Fk = F(x)
    P_pred = Fk @ P @ Fk.T + Q
    # update
    Hk = H(x_pred)
    y = z - h(x_pred)
    S = Hk @ P_pred @ Hk.T + R
    K = P_pred @ Hk.T @ np.linalg.inv(S)
    x_new = x_pred + K @ y
    P_new = (np.eye(len(x)) - K @ Hk) @ P_pred
    return x_new, P_new


__all__ = ["ekf_step"]
