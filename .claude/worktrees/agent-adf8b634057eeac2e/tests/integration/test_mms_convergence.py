"""L2 — Code verification: MMS + observed-order tests for core solvers.

Each test exercises a solver at 3 successive grid resolutions and checks the
observed convergence rate using ``naviertwin.core.verification`` helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from naviertwin.core.verification.loglog_slope import slope_fit
from naviertwin.core.verification.order_table import order_table

pytestmark = pytest.mark.convergence


def _u_exact_poisson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def _f_poisson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # -Δu = 2 π² sin(πx) sin(πy)
    return 2.0 * (np.pi ** 2) * np.sin(np.pi * x) * np.sin(np.pi * y)


class TestMultigridPoisson:
    """v_cycle_poisson should be ~2nd order in L2 error w.r.t. h."""

    def test_observed_order(self) -> None:
        from naviertwin.core.linalg.multigrid import v_cycle_poisson

        ns = [33, 65, 129]
        hs = []
        errs = []
        for n in ns:
            h = 1.0 / (n - 1)
            x = np.linspace(0, 1, n)
            X, Y = np.meshgrid(x, x, indexing="ij")
            u = np.zeros_like(X)
            f = _f_poisson(X, Y)
            for _ in range(40):
                u = v_cycle_poisson(u, f, h, n_pre=2, n_post=2, levels=4)
            err = float(np.sqrt(np.mean((u - _u_exact_poisson(X, Y)) ** 2)))
            hs.append(h)
            errs.append(err)

        p = slope_fit(np.array(hs), np.array(errs))
        # 2nd order ± 0.5 (multigrid + finite iter loosens slope)
        assert 1.5 <= p <= 2.5, f"observed order {p} out of band"


class TestADIHeatDecay:
    """ADI 2D heat: spatial error converges as h² (dt fixed, dt-error << h-error)."""

    def test_spatial_order(self) -> None:
        from naviertwin.core.solvers.adi_2d import adi_step

        nu = 1.0
        T = 0.02
        dt = 1e-4  # well below CFL spatial error to isolate h-order

        def run(n: int) -> float:
            h = 1.0 / (n - 1)
            x = np.linspace(0, 1, n)
            X, Y = np.meshgrid(x, x, indexing="ij")
            u = np.sin(np.pi * X) * np.sin(np.pi * Y)
            steps = int(round(T / dt))
            for _ in range(steps):
                u = adi_step(u, dt=dt, dx=h, dy=h, alpha=nu)
            u_exact = np.exp(-2 * (np.pi ** 2) * nu * (steps * dt)) * (
                np.sin(np.pi * X) * np.sin(np.pi * Y)
            )
            return float(np.sqrt(np.mean((u - u_exact) ** 2)))

        ns = [17, 33, 65]
        hs = np.array([1.0 / (n - 1) for n in ns])
        errs = np.array([run(n) for n in ns])
        t = order_table(hs, errs)
        # spatial 2nd order; allow ±0.5
        assert all(1.5 <= p <= 2.5 for p in t["p_pair"]), t["p_pair"]


class TestSpectralFourierExact:
    """Fourier exact heat step recovers analytic decay to machine precision."""

    def test_machine_precision(self) -> None:
        from naviertwin.core.solvers.spectral_galerkin import heat_step_fourier

        n = 64
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        u0 = np.sin(3 * x)
        dt = 0.05
        u = heat_step_fourier(u0, dt=dt, nu=1.0, L=2 * np.pi)
        u_exact = np.exp(-9 * dt) * np.sin(3 * x)
        assert np.max(np.abs(u - u_exact)) < 1e-10


class TestSSPRK3Decay:
    """SSP-RK3 on u' = -u → 3rd-order error in dt."""

    def test_third_order(self) -> None:
        from naviertwin.core.solvers.ssp_rk3 import ssp_rk3_step

        T = 0.5
        u_exact = np.exp(-T)

        def run(n: int) -> float:
            dt = T / n
            u = np.array([1.0])
            for _ in range(n):
                u = ssp_rk3_step(u, lambda u: -u, dt=dt)
            return float(abs(u[0] - u_exact))

        ns = [20, 40, 80]
        hs = np.array([T / n for n in ns])
        errs = np.array([run(n) for n in ns])
        p = slope_fit(hs, errs)
        # SSP-RK3 is 3rd-order; allow ±0.5
        assert 2.5 <= p <= 4.0, f"observed order {p}"


class TestWENO5Smooth:
    """WENO5 reconstruction at smooth point matches polynomial value."""

    def test_smooth_recon_high_accuracy(self) -> None:
        from naviertwin.core.solvers.weno5 import weno5_recon_left

        # Take 5-point stencil of f(x)=sin(x) at uniform spacing h
        def stencil(h: float) -> float:
            xs = np.array([0, h, 2 * h, 3 * h, 4 * h])
            u = np.sin(xs)
            v = weno5_recon_left(u)
            # exact value of sin at x = 2h + 0.5 h = 2.5 h
            return float(abs(v - np.sin(2.5 * h)))

        hs = np.array([0.1, 0.05, 0.025])
        errs = np.array([stencil(h) for h in hs])
        # high-order: errors decrease rapidly
        assert errs[1] < errs[0]
        assert errs[2] < errs[1]
