"""Round 270 — A category milestone: numerical methods import + e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneA:
    def test_imports(self) -> None:
        from naviertwin.core.linalg import (  # noqa: F401
            arnoldi,
            bicgstab,
            lanczos,
            minres,
            multigrid,
        )
        from naviertwin.core.solvers import (  # noqa: F401
            adi_2d,
            cheb_2d,
            imex_rk,
            spectral_galerkin,
        )

    def test_poisson_bicgstab_e2e(self) -> None:
        """Solve -Δu = 1 (1D, n=33) via BiCGStab on the discrete Laplacian."""
        from naviertwin.core.linalg.bicgstab import bicgstab

        n = 33
        h = 1.0 / (n - 1)
        A = (
            -2 * np.eye(n - 2)
            + np.diag(np.ones(n - 3), 1)
            + np.diag(np.ones(n - 3), -1)
        ) / (h * h)
        b = -np.ones(n - 2)
        x, info = bicgstab(A, b, tol=1e-8, max_iter=500)
        # exact: u(x) = 0.5 x (1-x); peak at center = 0.125
        assert info["converged"]
        assert np.isclose(x[(n - 2) // 2], 0.125, atol=1e-4)

    def test_arnoldi_ritz_diag(self) -> None:
        from naviertwin.core.linalg.arnoldi import ritz_values

        A = np.diag([4.0, 3.0, 2.0, 1.0])
        rv = ritz_values(A, np.ones(4), k=4)
        rv_sorted = sorted(rv.real)
        assert np.allclose(rv_sorted, [1.0, 2.0, 3.0, 4.0], atol=1e-6)

    def test_fourier_heat_step_e2e(self) -> None:
        """Fourier exact heat step matches analytic decay."""
        from naviertwin.core.solvers.spectral_galerkin import heat_step_fourier

        x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        u0 = np.sin(3 * x)
        u_t = heat_step_fourier(u0, dt=0.05, nu=1.0, L=2 * np.pi)
        # k=3, decay = e^{-9*0.05}
        assert np.allclose(u_t, np.exp(-0.45) * np.sin(3 * x), atol=1e-10)
