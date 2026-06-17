"""Unscented Kalman Filter (Julier & Uhlmann 1997).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_assimilation.ukf import ukf_step
    >>> x, P = np.array([0.0]), np.array([[1.0]])
    >>> def f(x): return x
    >>> def h(x): return x
    >>> x2, P2 = ukf_step(x, P, f, h, z=np.array([1.0]),
    ...                    Q=np.array([[0.01]]), R=np.array([[0.1]]))
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def _sigma_points(x: NDArray, P: NDArray, alpha: float = 1e-3,
                  kappa: float = 0.0) -> tuple[NDArray, NDArray, NDArray]:
    n = x.shape[0]
    lam = alpha * alpha * (n + kappa) - n
    sqrtP = np.linalg.cholesky((n + lam) * (P + 1e-12 * np.eye(n)))
    pts = np.zeros((2 * n + 1, n))
    pts[0] = x
    offsets = sqrtP.T
    pts[1 : 1 + n] = x[None, :] + offsets
    pts[1 + n :] = x[None, :] - offsets
    Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    Wm[0] = lam / (n + lam)
    Wc = Wm.copy()
    Wc[0] += 1 - alpha * alpha + 2.0
    return pts, Wm, Wc


def ukf_step(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    f: Callable[[NDArray], NDArray],
    h: Callable[[NDArray], NDArray],
    *,
    z: NDArray[np.float64],
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pts, Wm, Wc = _sigma_points(x, P)
    # propagate
    pts_p = np.stack(tuple(map(f, pts)))
    x_pred = (Wm[:, None] * pts_p).sum(axis=0)
    diff = pts_p - x_pred
    P_pred = (Wc[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0) + Q
    # measurement
    pts_m = np.stack(tuple(map(h, pts_p)))
    z_pred = (Wm[:, None] * pts_m).sum(axis=0)
    diff_z = pts_m - z_pred
    S = (Wc[:, None, None] * diff_z[:, :, None] * diff_z[:, None, :]).sum(axis=0) + R
    Pxz = (Wc[:, None, None] * diff[:, :, None] * diff_z[:, None, :]).sum(axis=0)
    K = Pxz @ np.linalg.inv(S)
    x_new = x_pred + K @ (z - z_pred)
    P_new = P_pred - K @ S @ K.T
    return x_new, P_new


__all__ = ["ukf_step"]
