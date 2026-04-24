"""Round 273 — GNAT."""

from __future__ import annotations

import numpy as np


class TestGNAT:
    def test_solution_in_basis(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.gnat import gnat_solve

        rng = np.random.default_rng(0)
        A = rng.standard_normal((6, 6)) + 6 * np.eye(6)
        b = rng.standard_normal(6)
        x_true = np.linalg.solve(A, b)
        Q, _ = np.linalg.qr(np.column_stack([x_true, rng.standard_normal((6, 2))]))
        x = gnat_solve(A, b, Q, n_samples=3)
        # x_true ∈ span(Q) → reconstruction should be close
        assert np.linalg.norm(x - x_true) < 1e-6

    def test_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.gnat import gnat_solve

        rng = np.random.default_rng(2)
        A = rng.standard_normal((10, 10))
        b = rng.standard_normal(10)
        Phi = np.linalg.qr(rng.standard_normal((10, 4)))[0]
        x = gnat_solve(A, b, Phi)
        assert x.shape == (10,)
