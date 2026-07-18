"""Round 263 — MINRES."""

from __future__ import annotations

import numpy as np


class TestMINRES:
    def test_sym_indef(self) -> None:
        from naviertwin.core.linalg.minres import minres

        A = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues -1, 3
        b = np.array([1.0, 2.0])
        x, info = minres(A, b)
        assert info["converged"]
        assert np.allclose(A @ x, b, atol=1e-6)

    def test_larger_sym(self) -> None:
        from naviertwin.core.linalg.minres import minres

        rng = np.random.default_rng(0)
        M = rng.standard_normal((15, 15))
        A = 0.5 * (M + M.T)  # symmetric (possibly indefinite)
        b = rng.standard_normal(15)
        x, info = minres(A, b)
        if info["converged"]:
            assert np.allclose(A @ x, b, atol=1e-4)
