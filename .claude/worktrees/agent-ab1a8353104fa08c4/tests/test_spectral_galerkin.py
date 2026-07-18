"""Round 268 — Spectral Galerkin Fourier."""

from __future__ import annotations

import numpy as np


class TestSpectral:
    def test_diff_sin(self) -> None:
        from naviertwin.core.solvers.spectral_galerkin import fourier_diff

        x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        u = np.sin(x)
        ux = fourier_diff(u, order=1, L=2 * np.pi)
        assert np.allclose(ux, np.cos(x), atol=1e-10)

    def test_diff2_sin(self) -> None:
        from naviertwin.core.solvers.spectral_galerkin import fourier_diff

        x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        u = np.sin(2 * x)
        uxx = fourier_diff(u, order=2, L=2 * np.pi)
        assert np.allclose(uxx, -4 * np.sin(2 * x), atol=1e-9)

    def test_heat_decay(self) -> None:
        from naviertwin.core.solvers.spectral_galerkin import heat_step_fourier

        x = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        u = np.sin(x)
        u_t = heat_step_fourier(u, dt=0.1, nu=1.0, L=2 * np.pi)
        # exact: e^{-0.1} sin(x) (k=1, ν k²=1)
        assert np.allclose(u_t, np.exp(-0.1) * np.sin(x), atol=1e-10)
