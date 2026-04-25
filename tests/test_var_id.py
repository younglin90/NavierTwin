"""Round 423 — VAR identification."""

from __future__ import annotations

import numpy as np


class TestVAR:
    def test_recovers_var1(self) -> None:
        from naviertwin.core.system_id.var_id import fit_var

        rng = np.random.default_rng(0)
        A_true = np.array([[0.5, 0.1], [-0.05, 0.7]])
        x = [rng.standard_normal(2)]
        for _ in range(500):
            x.append(A_true @ x[-1] + 0.001 * rng.standard_normal(2))
        X = np.asarray(x)
        As = fit_var(X, p=1)
        assert np.allclose(As[0], A_true, atol=0.05)

    def test_p2_shape(self) -> None:
        from naviertwin.core.system_id.var_id import fit_var

        X = np.random.default_rng(0).standard_normal((50, 3))
        As = fit_var(X, p=2)
        assert len(As) == 2
        assert all(A.shape == (3, 3) for A in As)
