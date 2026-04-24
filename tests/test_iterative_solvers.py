"""Round 116 — Jacobi / Gauss-Seidel / CG."""

from __future__ import annotations

import numpy as np


def _spd(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)
    b = rng.standard_normal(n)
    return A, b


class TestSolvers:
    def test_jacobi_converges(self) -> None:
        from naviertwin.core.linalg.iterative_solvers import jacobi

        A = np.array([[10.0, 1.0], [1.0, 10.0]])
        b = np.array([11.0, 11.0])
        x, info = jacobi(A, b, max_iter=200)
        assert info["converged"]
        assert np.allclose(A @ x, b, atol=1e-6)

    def test_gauss_seidel(self) -> None:
        from naviertwin.core.linalg.iterative_solvers import gauss_seidel

        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        x, info = gauss_seidel(A, b, max_iter=200)
        assert info["converged"]
        assert np.allclose(A @ x, b, atol=1e-6)

    def test_cg_spd(self) -> None:
        from naviertwin.core.linalg.iterative_solvers import conjugate_gradient

        A, b = _spd(20, seed=0)
        x, info = conjugate_gradient(A, b)
        assert info["converged"]
        assert np.allclose(A @ x, b, atol=1e-6)

    def test_cg_vs_direct(self) -> None:
        from naviertwin.core.linalg.iterative_solvers import conjugate_gradient

        A, b = _spd(30, seed=1)
        x_cg, _ = conjugate_gradient(A, b)
        x_direct = np.linalg.solve(A, b)
        assert np.allclose(x_cg, x_direct, atol=1e-6)
