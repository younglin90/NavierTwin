"""Round 472 — SPH momentum."""

from __future__ import annotations

import numpy as np


class TestSPHMom:
    def test_uniform_no_acceleration(self) -> None:
        from naviertwin.core.meshless.sph_momentum import sph_acceleration_1d

        x = np.linspace(0, 1, 11)
        m = np.full(11, 0.1)
        rho = np.full(11, 1.0)
        p = np.full(11, 1.0)
        a = sph_acceleration_1d(x, m, rho, p, h=0.3)
        # interior: pressure uniform, symmetric → ~0
        assert abs(a[5]) < 0.1

    def test_pressure_gradient_pushes(self) -> None:
        from naviertwin.core.meshless.sph_momentum import sph_acceleration_1d

        x = np.linspace(0, 1, 11)
        m = np.full(11, 0.1)
        rho = np.full(11, 1.0)
        p = np.linspace(2.0, 0.0, 11)  # high p on left → push right
        a = sph_acceleration_1d(x, m, rho, p, h=0.3)
        # interior: a > 0 (rightward)
        assert a[5] > 0
