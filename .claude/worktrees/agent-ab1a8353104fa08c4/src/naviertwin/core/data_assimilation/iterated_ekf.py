"""Iterated EKF — relinearize H around updated estimate.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_assimilation.iterated_ekf import iekf_step
    >>> x, P = np.zeros(1), np.eye(1)
    >>> x2, P2 = iekf_step(x, P, f=lambda x: x, F=lambda x: np.eye(1),
    ...                    h=lambda x: x**2, H=lambda x: 2*x[:, None].T,
    ...                    z=np.array([1.0]),
    ...                    Q=0.01*np.eye(1), R=0.01*np.eye(1))
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.linalg import solve as _np_solve
from numpy.typing import NDArray

from naviertwin._native import HAS_NATIVE_KERNELS, _kernels


def _right_solve(S: NDArray[np.float64], A: NDArray[np.float64]) -> NDArray[np.float64]:
    native_solve = (
        getattr(_kernels, "solve_dense", None)
        if HAS_NATIVE_KERNELS
        else None
    )
    if native_solve is not None and A.shape[0] == 1:
        return native_solve(S.T, A.ravel()).reshape(1, -1)
    return _np_solve(S.T, A.T).T


def iekf_step(
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
    n_iter: int = 5,
    tol: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # predict
    x_pred = f(x)
    Fk = F(x)
    P_pred = Fk @ P @ Fk.T + Q
    # iterate update
    x_iter = x_pred.copy()
    it = 0
    while it < n_iter:
        Hk = H(x_iter)
        S = Hk @ P_pred @ Hk.T + R
        K = _right_solve(S, P_pred @ Hk.T)
        innov = z - h(x_iter) - Hk @ (x_pred - x_iter)
        x_new = x_pred + K @ innov
        if np.linalg.norm(x_new - x_iter) < tol:
            x_iter = x_new
            break
        x_iter = x_new
        it += 1
    Hk = H(x_iter)
    P_new = (np.eye(len(x)) - K @ Hk) @ P_pred
    return x_iter, P_new


__all__ = ["iekf_step"]
