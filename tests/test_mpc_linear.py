"""Round 198 — Linear MPC."""

from __future__ import annotations

import numpy as np


class TestMPC:
    def test_scalar_setpoint(self) -> None:
        from naviertwin.core.control.mpc_linear import mpc_step

        A = np.array([[0.9]])
        B = np.array([[0.1]])
        C = np.array([[1.0]])
        x = np.array([0.0])
        N = 10
        ref = np.ones((N, 1))
        U = mpc_step(A, B, C, x, ref, Q=1.0, R=1e-3)
        # Forward simulate
        ys = []
        for k in range(N):
            x = A @ x + B @ U[k]
            ys.append((C @ x)[0])
        # 궁극적으로 참조값에 근접
        assert abs(ys[-1] - 1.0) < 0.1

    def test_regulation_to_zero(self) -> None:
        from naviertwin.core.control.mpc_linear import mpc_step

        A = np.array([[1.05]])  # unstable
        B = np.array([[1.0]])
        C = np.array([[1.0]])
        x = np.array([1.0])
        ref = np.zeros((8, 1))
        U = mpc_step(A, B, C, x, ref, Q=1.0, R=1e-2)
        # Forward simulate
        for k in range(8):
            x = A @ x + B @ U[k]
        assert abs(x[0]) < 0.3  # 0 으로 회귀
