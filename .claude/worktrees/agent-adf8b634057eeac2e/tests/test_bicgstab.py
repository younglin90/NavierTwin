"""Round 262 — BiCGStab."""

from __future__ import annotations

import numpy as np


class TestBiCG:
    def test_nonsymmetric(self) -> None:
        from naviertwin.core.linalg.bicgstab import bicgstab

        rng = np.random.default_rng(0)
        A = rng.standard_normal((20, 20)) + 10 * np.eye(20)
        b = rng.standard_normal(20)
        x, info = bicgstab(A, b, tol=1e-8)
        assert info["converged"]
        assert np.allclose(A @ x, b, atol=1e-6)

    def test_matches_direct(self) -> None:
        from naviertwin.core.linalg.bicgstab import bicgstab

        rng = np.random.default_rng(1)
        A = rng.standard_normal((30, 30)) + 15 * np.eye(30)
        b = rng.standard_normal(30)
        x_bi, _ = bicgstab(A, b, tol=1e-10)
        x_dir = np.linalg.solve(A, b)
        assert np.allclose(x_bi, x_dir, atol=1e-6)

    def test_callable_A(self) -> None:
        from naviertwin.core.linalg.bicgstab import bicgstab

        A = np.array([[4.0, 1.0], [2.0, 3.0]])
        b = np.array([1.0, 2.0])
        x, info = bicgstab(lambda v: A @ v, b, tol=1e-10)
        assert info["converged"]
        assert np.allclose(A @ x, b, atol=1e-6)
