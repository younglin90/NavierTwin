"""Round 346 — LQG."""

from __future__ import annotations

import numpy as np


class TestLQG:
    def test_stabilizes_double_integrator(self) -> None:
        from naviertwin.core.control.lqg import LQGController

        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        C = np.array([[1.0, 0.0]])
        ctrl = LQGController(
            A, B, C,
            Q=np.eye(2), R=np.eye(1),
            Qw=0.01 * np.eye(2), Rv=0.1 * np.eye(1),
        )
        x = np.array([5.0, 0.0])
        rng = np.random.default_rng(0)
        for _ in range(50):
            y = C @ x + rng.normal(0, 0.05, size=1)
            u = ctrl.step(y)
            x = A @ x + B @ u
        # state regulated to near 0
        assert np.linalg.norm(x) < 1.0
