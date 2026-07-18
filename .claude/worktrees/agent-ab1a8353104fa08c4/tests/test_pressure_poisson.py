"""Round 142 — 2D Poisson solvers."""

from __future__ import annotations

import numpy as np


class TestPoisson:
    def test_fft_periodic(self) -> None:
        """f = -2π² sin(πx)sin(πy) → p = sin(πx)sin(πy) (on [0,2] × [0,2] periodic)."""
        from naviertwin.core.solvers.pressure_poisson import poisson_2d_fft

        nx, ny = 64, 64
        Lx = Ly = 2.0
        x = np.linspace(0, Lx, nx, endpoint=False)
        y = np.linspace(0, Ly, ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        p_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
        f = -2 * (np.pi ** 2) * p_true
        p = poisson_2d_fft(f, Lx, Ly)
        # mean 조정 후 비교
        err = float(np.max(np.abs(p - p_true)))
        assert err < 1e-6

    def test_jacobi_dirichlet(self) -> None:
        """sin(πx/L) sin(πy/L) Dirichlet 해."""
        from naviertwin.core.solvers.pressure_poisson import poisson_2d_jacobi

        nx = ny = 32
        L = 1.0
        x = np.linspace(0, L, nx)
        y = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        p_true = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
        f = -2 * (np.pi / L) ** 2 * p_true
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        p, info = poisson_2d_jacobi(f, dx, dy, max_iter=20000, tol=1e-5)
        assert info["converged"]
        assert float(np.max(np.abs(p - p_true))) < 0.05
