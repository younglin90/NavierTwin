"""LQG — LQR + Kalman filter (separation principle).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.control.lqg import LQGController
    >>> A = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> B = np.array([[0.0], [1.0]])
    >>> C = np.array([[1.0, 0.0]])
    >>> Q, R = np.eye(2), np.eye(1)
    >>> Qw, Rv = 0.01*np.eye(2), 0.1*np.eye(1)
    >>> ctrl = LQGController(A, B, C, Q, R, Qw, Rv)
    >>> u = ctrl.step(np.array([1.0]))
    >>> u.shape
    (1,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.control.lqr import lqr_gain
from naviertwin.core.data_assimilation.iterated_ekf import _right_solve


class LQGController:
    def __init__(
        self,
        A: NDArray, B: NDArray, C: NDArray,
        Q: NDArray, R: NDArray, Qw: NDArray, Rv: NDArray,
    ) -> None:
        self.A = A
        self.B = B
        self.C = C
        self.K = lqr_gain(A, B, Q, R)
        # Kalman gain via dual DARE (P = A P Aᵀ - A P Cᵀ (C P Cᵀ + Rv)⁻¹ C P Aᵀ + Qw)
        # iterative
        P = Qw.copy()
        it = 0
        while it < 200:
            S = C @ P @ C.T + Rv
            L = _right_solve(S, A @ P @ C.T)
            P_new = A @ P @ A.T - L @ S @ L.T + Qw
            if np.max(np.abs(P_new - P)) < 1e-10:
                P = P_new
                break
            P = P_new
            it += 1
        self.L = L
        self.x_hat = np.zeros(A.shape[0])

    def step(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        u = -self.K @ self.x_hat
        # update estimator
        self.x_hat = self.A @ self.x_hat + self.B @ u + self.L @ (y - self.C @ self.x_hat)
        return u


__all__ = ["LQGController"]
