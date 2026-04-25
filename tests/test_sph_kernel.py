"""Round 471 — SPH kernel."""

from __future__ import annotations

import numpy as np


class TestSPH:
    def test_kernel_normalized(self) -> None:
        from naviertwin.core.meshless.sph_kernel import cubic_spline_1d

        r = np.linspace(-3, 3, 1001)
        w = cubic_spline_1d(r, h=1.0)
        # ∫ W dr ≈ 1
        integral = float(np.trapezoid(w, r))
        assert abs(integral - 1.0) < 0.01

    def test_density_uniform(self) -> None:
        from naviertwin.core.meshless.sph_kernel import density_1d

        x = np.linspace(0, 10, 21)
        m = np.full_like(x, 0.5)
        rho = density_1d(x, m, h=1.0)
        # interior density roughly constant
        assert rho[10] > 0
        assert (rho > 0).all()
