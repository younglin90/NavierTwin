"""Round 345 — LQR."""

from __future__ import annotations

import numpy as np


class TestLQR:
    def test_stabilizes(self) -> None:
        from naviertwin.core.control.lqr import lqr_gain

        A = np.array([[1.2, 0.1], [0.0, 1.0]])  # unstable
        B = np.array([[0.0], [0.1]])
        Q = np.eye(2)
        R = np.eye(1)
        K = lqr_gain(A, B, Q, R)
        # closed-loop A - B K should be stable (|eig|<1)
        eigs = np.linalg.eigvals(A - B @ K)
        assert (np.abs(eigs) < 1.0).all()

    def test_shape(self) -> None:
        from naviertwin.core.control.lqr import lqr_gain

        A = np.eye(3)
        B = np.array([[1.0], [0.0], [0.0]])
        K = lqr_gain(A, B, np.eye(3), np.eye(1))
        assert K.shape == (1, 3)
