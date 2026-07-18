"""Round 206 — PCG."""

from __future__ import annotations

import numpy as np


class TestPCG:
    def test_spd(self) -> None:
        from naviertwin.core.linalg.pcg import pcg

        rng = np.random.default_rng(0)
        M = rng.standard_normal((20, 20))
        A = M @ M.T + 20 * np.eye(20)
        b = rng.standard_normal(20)
        x, info = pcg(A, b)
        assert info["converged"]
        assert np.allclose(A @ x, b, atol=1e-6)

    def test_vs_direct(self) -> None:
        from naviertwin.core.linalg.pcg import pcg

        rng = np.random.default_rng(1)
        M = rng.standard_normal((30, 30))
        A = M @ M.T + 30 * np.eye(30)
        b = rng.standard_normal(30)
        x_pcg, _ = pcg(A, b)
        x_dir = np.linalg.solve(A, b)
        assert np.allclose(x_pcg, x_dir, atol=1e-6)

    def test_zero_diag(self) -> None:
        import pytest

        from naviertwin.core.linalg.pcg import jacobi_preconditioner

        with pytest.raises(ValueError):
            jacobi_preconditioner(np.zeros((2, 2)))
