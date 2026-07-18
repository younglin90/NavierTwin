"""Round 404 — operator inference."""

from __future__ import annotations

import numpy as np


class TestOpInf:
    def test_linear_recover(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.opinf import opinf_fit

        rng = np.random.default_rng(0)
        A_true = np.diag([-2.0, -1.0])
        Z = rng.standard_normal((200, 2))
        Zdot = Z @ A_true.T
        ops = opinf_fit(Z, Zdot)
        assert ops["A"].shape == (2, 2)
        assert np.allclose(ops["A"], A_true, atol=1e-8)

    def test_with_input(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.opinf import opinf_fit

        rng = np.random.default_rng(1)
        Z = rng.standard_normal((100, 2))
        U = rng.standard_normal((100, 1))
        Zdot = -Z + 0.5 * U
        ops = opinf_fit(Z, Zdot, U=U)
        assert ops["B"].shape == (2, 1)
        assert np.allclose(ops["B"], 0.5, atol=0.1)
