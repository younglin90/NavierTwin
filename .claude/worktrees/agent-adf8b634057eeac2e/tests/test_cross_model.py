"""Round 484 — Cross."""

from __future__ import annotations


class TestCross:
    def test_zero_shear(self) -> None:
        from naviertwin.core.rheology.cross_model import cross_viscosity

        assert cross_viscosity(gamma_dot=0, mu_0=5, mu_inf=0.1, k=1, m=1) == 5.0

    def test_high_shear(self) -> None:
        from naviertwin.core.rheology.cross_model import cross_viscosity

        v = cross_viscosity(gamma_dot=1e6, mu_0=5, mu_inf=0.1, k=1, m=1)
        assert abs(v - 0.1) < 1e-3
