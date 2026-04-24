"""Round 408 — closure ROM."""

from __future__ import annotations

import numpy as np


class TestClosureROM:
    def test_recovers_K(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.closure_rom import (
            apply_closure,
            fit_closure,
        )

        rng = np.random.default_rng(0)
        K_true = np.array([[0.1, -0.2], [0.05, 0.3]])
        Z = rng.standard_normal((100, 2))
        R = (Z @ K_true.T)
        K = fit_closure(Z, R)
        assert np.allclose(K, K_true, atol=1e-9)
        # apply
        z = np.array([1.0, 2.0])
        assert np.allclose(apply_closure(K, z), K_true @ z)
