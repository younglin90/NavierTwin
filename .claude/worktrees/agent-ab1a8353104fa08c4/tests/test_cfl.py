"""Round 194 — CFL."""

from __future__ import annotations

import numpy as np
import pytest


class TestCFL:
    def test_convective(self) -> None:
        from naviertwin.core.solvers.cfl import cfl_convective

        assert cfl_convective(0.01, 2.0, cfl=0.8) == pytest.approx(0.004)

    def test_diffusive(self) -> None:
        from naviertwin.core.solvers.cfl import cfl_diffusive

        assert cfl_diffusive(0.01, 0.01, cfl=0.5) == pytest.approx(0.0025)

    def test_combined(self) -> None:
        from naviertwin.core.solvers.cfl import cfl_combined

        dt = cfl_combined(0.01, u_max=1.0, nu=1e-3)
        assert dt > 0

    def test_cfl_number(self) -> None:
        from naviertwin.core.solvers.cfl import cfl_number

        assert cfl_number(dt=0.01, dx=0.1, u_max=5.0) == pytest.approx(0.5)

    def test_cfl_field(self) -> None:
        from naviertwin.core.solvers.cfl import cfl_field

        u = np.ones((5, 5)) * 2.0
        v = np.zeros((5, 5))
        c = cfl_field(dt=0.01, dx=0.1, dy=0.1, u=u, v=v)
        assert abs(c - 0.2) < 1e-12

    def test_zero_division(self) -> None:
        from naviertwin.core.solvers.cfl import cfl_convective, cfl_diffusive

        assert cfl_convective(0.1, 0.0) == float("inf")
        assert cfl_diffusive(0.1, 0.0) == float("inf")
