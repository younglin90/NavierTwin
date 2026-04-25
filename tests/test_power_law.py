"""Round 481 — power-law rheology."""

from __future__ import annotations


class TestPowerLaw:
    def test_newtonian_n1(self) -> None:
        from naviertwin.core.rheology.power_law import apparent_viscosity

        assert abs(apparent_viscosity(gamma_dot=2.0, K=0.5, n=1.0) - 0.5) < 1e-12

    def test_shear_thinning(self) -> None:
        from naviertwin.core.rheology.power_law import apparent_viscosity

        # n<1: viscosity decreases as γ̇ increases
        v1 = apparent_viscosity(gamma_dot=0.1, K=1.0, n=0.5)
        v2 = apparent_viscosity(gamma_dot=10.0, K=1.0, n=0.5)
        assert v1 > v2

    def test_stress(self) -> None:
        from naviertwin.core.rheology.power_law import shear_stress

        # τ = K γ̇^n (positive)
        s = shear_stress(gamma_dot=4.0, K=1.0, n=0.5)
        assert abs(s - 2.0) < 1e-12
