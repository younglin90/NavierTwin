"""Round 172 — DMDc."""

from __future__ import annotations

import numpy as np


class TestDMDc:
    def test_recover_linear_controlled_system(self) -> None:
        from naviertwin.core.system_id.dmdc import fit_dmdc, rollout_dmdc

        rng = np.random.default_rng(0)
        A_true = np.array([[0.9, 0.1], [-0.2, 0.8]])
        B_true = np.array([[1.0], [0.5]])
        T = 200
        X = np.zeros((2, T))
        U = rng.standard_normal((1, T - 1))
        for k in range(T - 1):
            X[:, k + 1] = A_true @ X[:, k] + B_true[:, 0] * U[0, k]
        A, B = fit_dmdc(X, U)
        assert np.allclose(A, A_true, atol=1e-8)
        assert np.allclose(B, B_true, atol=1e-8)

        # rollout 매칭
        x0 = X[:, 0]
        pred = rollout_dmdc(A, B, x0, U)
        assert np.allclose(pred, X, atol=1e-6)

    def test_shape_errors(self) -> None:
        import pytest as _pt

        from naviertwin.core.system_id.dmdc import fit_dmdc

        X = np.zeros((3, 10))
        U = np.zeros((1, 5))
        with _pt.raises(ValueError):
            fit_dmdc(X, U)
