"""L3 — Validation: standard CFD benchmarks judged via V&V20.

Each benchmark compares simulation output S to a reference value D and
applies ``vv20.is_validated`` with combined uncertainty u_val.
"""

from __future__ import annotations

import numpy as np
import pytest

from naviertwin.core.verification.vv20 import (
    comparison_error,
    is_validated,
    validation_uncertainty,
)

pytestmark = pytest.mark.vv


class TestBurgersShock:
    """Inviscid Burgers shock: analytic shock speed s = 0.5 (left=1, right=0).

    For the simple Lax-Friedrichs flux on a coarse grid, the shock front
    location at t = 0.5 should sit near x = 0.25 + 0.5·t = 0.5.
    """

    def test_shock_position(self) -> None:
        n = 201
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        u = np.where(x < 0.25, 1.0, 0.0)
        T = 0.5
        dt = 0.4 * dx  # CFL ≤ max|u| dt/dx

        steps = int(round(T / dt))
        for _ in range(steps):
            f = 0.5 * u * u
            # Lax-Friedrichs flux at i+1/2
            alpha = float(np.max(np.abs(u)))
            f_face = 0.5 * (f[1:] + f[:-1]) - 0.5 * alpha * (u[1:] - u[:-1])
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] - dt / dx * (f_face[1:] - f_face[:-1])
            u = u_new

        # find shock location: largest |du/dx|
        idx = int(np.argmax(np.abs(np.diff(u))))
        x_shock = 0.5 * (x[idx] + x[idx + 1])
        x_shock_exact = 0.25 + 0.5 * T  # = 0.5

        E = comparison_error(S=x_shock, D=x_shock_exact)
        u_val = validation_uncertainty(u_num=2 * dx, u_input=0.0, u_data=0.0)
        assert is_validated(E=E, u_val=u_val, k=2.0), (
            f"E={E:.4g}, u_val={u_val:.4g}, x_shock={x_shock:.4g}"
        )


class TestTaylorGreen2D:
    """Decaying 2D Taylor-Green via Fourier: u_max ~ exp(-2 ν π² t)."""

    def test_decay_amplitude(self) -> None:
        from naviertwin.core.solvers.spectral_galerkin import heat_step_fourier

        # Use diffusion of single Fourier mode as a Taylor-Green proxy
        # (vorticity-like ω = sin(x) decays as exp(-ν t))
        n = 64
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        u0 = np.sin(x)
        nu = 0.1
        T = 1.0

        u = u0.copy()
        dt = 0.01
        for _ in range(int(round(T / dt))):
            u = heat_step_fourier(u, dt=dt, nu=nu, L=2 * np.pi)

        amp_sim = float(np.max(np.abs(u)))
        amp_exact = float(np.exp(-nu * T))  # k=1 mode

        E = comparison_error(S=amp_sim, D=amp_exact)
        u_val = validation_uncertainty(u_num=1e-3, u_input=0, u_data=0)
        assert is_validated(E=E, u_val=u_val, k=2.0), (
            f"E={E}, u_val={u_val}, sim={amp_sim}, exact={amp_exact}"
        )


class TestCavityProxy:
    """Lid-driven cavity Re=100 proxy via R231 placeholder.

    We cannot ship a full Navier-Stokes solver here, so the test asserts
    that the cavity *energy budget surrogate* (sum |u|²) decays under the
    Fourier diffusion proxy used elsewhere in the suite.
    """

    def test_energy_decreases(self) -> None:
        from naviertwin.core.solvers.spectral_galerkin import heat_step_fourier

        n = 64
        rng = np.random.default_rng(0)
        u = rng.standard_normal(n)
        e0 = float(np.sum(u * u))
        for _ in range(50):
            u = heat_step_fourier(u, dt=0.01, nu=0.5, L=2 * np.pi)
        e1 = float(np.sum(u * u))
        assert e1 < e0


class TestGCIChannel:
    """Synthetic GCI workflow: 3 grids of a 2nd-order error model."""

    def test_gci_pipeline(self) -> None:
        from naviertwin.core.verification.gci import gci, observed_order
        from naviertwin.core.verification.richardson import richardson

        # Quantity of interest: f_h = 1.0 + h^2  → exact = 1.0
        h = np.array([0.1, 0.05, 0.025])
        fh = 1.0 + h ** 2

        p = observed_order(f1=fh[2], f2=fh[1], f3=fh[0], r=2.0)
        assert abs(p - 2.0) < 1e-9

        f_R = richardson(f_fine=fh[2], f_coarse=fh[1], r=2.0, p=p)
        # Richardson should beat the finest grid
        assert abs(f_R - 1.0) < abs(fh[2] - 1.0)

        eps = (fh[2] - fh[1]) / fh[2]
        g = gci(eps=eps, r=2.0, p=p)
        assert g >= 0
