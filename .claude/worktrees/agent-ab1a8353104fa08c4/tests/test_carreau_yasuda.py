"""Round 483 — Carreau-Yasuda."""

from __future__ import annotations


class TestCY:
    def test_zero_shear(self) -> None:
        from naviertwin.core.rheology.carreau_yasuda import cy_viscosity

        # γ̇ = 0 → μ_0
        v = cy_viscosity(gamma_dot=0.0, mu_0=10, mu_inf=1, lam=1, a=2, n=0.5)
        assert abs(v - 10.0) < 1e-12

    def test_high_shear_approaches_inf(self) -> None:
        from naviertwin.core.rheology.carreau_yasuda import cy_viscosity

        v = cy_viscosity(gamma_dot=1e6, mu_0=10, mu_inf=1, lam=1, a=2, n=0.5)
        assert abs(v - 1.0) < 0.01
