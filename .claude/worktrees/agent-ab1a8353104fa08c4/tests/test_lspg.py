"""Round 271 — LSPG."""

from __future__ import annotations

import numpy as np


class TestLSPG:
    def test_full_basis_recovers(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.lspg import lspg_solve

        rng = np.random.default_rng(0)
        A = rng.standard_normal((5, 5)) + 5 * np.eye(5)
        b = rng.standard_normal(5)
        x = lspg_solve(A, b, np.eye(5))
        assert np.allclose(x, np.linalg.solve(A, b), atol=1e-8)

    def test_solution_basis_zero_residual(self) -> None:
        """If Phi contains the true solution, LSPG residual = 0."""
        from naviertwin.core.dimensionality_reduction.linear.lspg import (
            lspg_residual_norm,
        )

        rng = np.random.default_rng(1)
        A = rng.standard_normal((5, 5)) + 5 * np.eye(5)
        b = rng.standard_normal(5)
        x_true = np.linalg.solve(A, b)
        # Phi spans true x + a perpendicular direction
        Q, _ = np.linalg.qr(np.column_stack([x_true, rng.standard_normal(5)]))
        r = lspg_residual_norm(A, b, Q)
        assert r < 1e-8
