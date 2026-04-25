"""Round 421 — nonlinear KS."""

from __future__ import annotations

import numpy as np


class TestKSNonlinear:
    def test_smoother_runs(self) -> None:
        from naviertwin.core.data_assimilation.ks_nonlinear import nonlinear_rts

        N = 10
        xs = np.random.default_rng(0).standard_normal((N, 2))
        Ps = np.tile(np.eye(2), (N, 1, 1))
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        xs_s, Ps_s = nonlinear_rts(xs, Ps, F, Q=0.01 * np.eye(2))
        assert xs_s.shape == xs.shape
        assert Ps_s.shape == Ps.shape
        assert np.isfinite(xs_s).all()
