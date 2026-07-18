"""Round 217 — Newton-Krylov + GMRES."""

from __future__ import annotations

import numpy as np


class TestNK:
    def test_gmres_linear(self) -> None:
        from naviertwin.core.linalg.newton_krylov import gmres

        rng = np.random.default_rng(0)
        A = rng.standard_normal((10, 10)) + 5 * np.eye(10)
        b = rng.standard_normal(10)
        x = gmres(lambda v: A @ v, b, max_iter=20)
        assert np.linalg.norm(A @ x - b) < 1e-6

    def test_newton_krylov_system(self) -> None:
        from naviertwin.core.linalg.newton_krylov import newton_krylov

        def F(x):
            return np.array([x[0] ** 2 - 4.0, x[1] ** 3 - 8.0])

        x, info = newton_krylov(F, np.array([1.5, 1.5]), max_iter=50)
        assert info["converged"]
        assert abs(x[0] - 2.0) < 1e-4
        assert abs(x[1] - 2.0) < 1e-4
