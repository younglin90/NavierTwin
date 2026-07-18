"""Round 90 — 유체역학 무차원수."""

from __future__ import annotations

import pytest


class TestDimensionless:
    def test_reynolds(self) -> None:
        from naviertwin.core.analysis.dimensionless import reynolds, reynolds_rho

        assert reynolds(10.0, 0.1, 1e-5) == pytest.approx(1e5)
        assert reynolds_rho(1.0, 10.0, 1000.0, 1e-3) == pytest.approx(1e7)

    def test_mach_and_prandtl(self) -> None:
        from naviertwin.core.analysis.dimensionless import mach, prandtl

        assert mach(340.0, 340.0) == pytest.approx(1.0)
        assert prandtl(1005.0, 1.8e-5, 0.025) == pytest.approx(0.7236, rel=1e-3)

    def test_others(self) -> None:
        from naviertwin.core.analysis.dimensionless import (
            froude,
            nusselt,
            peclet,
            rayleigh,
            strouhal,
            weber,
        )

        assert nusselt(100.0, 0.1, 0.5) == pytest.approx(20.0)
        assert froude(10.0, 1.0) == pytest.approx(10.0 / (9.81 ** 0.5), rel=1e-6)
        assert weber(1000.0, 1.0, 0.1, 0.072) == pytest.approx(1000.0 / 0.72)
        assert peclet(2.0, 0.1, 1e-5) == pytest.approx(20000.0)
        assert strouhal(5.0, 0.1, 1.0) == pytest.approx(0.5)
        assert rayleigh(9.81, 3e-3, 10.0, 0.1, 1.5e-5, 2e-5) > 0
