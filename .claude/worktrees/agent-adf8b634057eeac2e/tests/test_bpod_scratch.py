"""Round 214 — BPOD scratch."""

from __future__ import annotations

import numpy as np


class TestBPOD:
    def test_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.bpod_scratch import (
            bpod_reduce,
        )

        A = np.diag([0.9, 0.8, 0.7, 0.6])
        B = np.ones((4, 1))
        C = np.ones((1, 4))
        Ar, Br, Cr, T, Tinv = bpod_reduce(A, B, C, r=2)
        assert Ar.shape == (2, 2)
        assert Br.shape == (2, 1)
        assert Cr.shape == (1, 2)

    def test_step_response_match(self) -> None:
        """B축소 시 step response 가 유사."""
        from naviertwin.core.dimensionality_reduction.linear.bpod_scratch import (
            bpod_reduce,
        )

        A = np.array([[0.9, 0.05], [0.1, 0.85]])
        B = np.array([[1.0], [0.5]])
        C = np.array([[1.0, 1.0]])
        Ar, Br, Cr, _, _ = bpod_reduce(A, B, C, r=2)

        # 단위 impulse 응답 비교
        x = np.zeros(2)
        xr = np.zeros(2)
        ys = []
        yrs = []
        for k in range(20):
            u = 1.0 if k == 0 else 0.0
            x = A @ x + B[:, 0] * u
            xr = Ar @ xr + Br[:, 0] * u
            ys.append(float((C @ x).ravel()[0]))
            yrs.append(float((Cr @ xr).ravel()[0]))
        assert np.allclose(ys, yrs, atol=1e-6)

    def test_lyapunov_solves(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.bpod_scratch import (
            lyapunov_disc,
        )

        A = np.array([[0.5, 0.1], [0.0, 0.4]])
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        X = lyapunov_disc(A, Q)
        res = X - A @ X @ A.T - Q
        assert np.max(np.abs(res)) < 1e-6
