"""Round 349 — MHE."""

from __future__ import annotations

import numpy as np


class TestMHE:
    def test_returns_finite(self) -> None:
        from naviertwin.core.data_assimilation.mhe import mhe_estimate

        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Y = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        x = mhe_estimate(
            A, H, Y, x0=np.array([0.0, 0.0]),
            P0=np.eye(2), Q=0.01 * np.eye(2), R=0.1 * np.eye(1),
        )
        assert x.shape == (2,)
        assert np.isfinite(x).all()

    def test_constant_state(self) -> None:
        """A=I, H=I; observations near 5 → estimate near 5."""
        from naviertwin.core.data_assimilation.mhe import mhe_estimate

        rng = np.random.default_rng(0)
        Y = 5.0 * np.ones((20, 1)) + 0.1 * rng.standard_normal((20, 1))
        x = mhe_estimate(
            np.eye(1), np.eye(1), Y, x0=np.array([0.0]),
            P0=np.eye(1), Q=0.01 * np.eye(1), R=0.01 * np.eye(1),
        )
        assert abs(x[0] - 5.0) < 0.5
