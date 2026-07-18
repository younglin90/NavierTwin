"""Round 482 — Bingham plastic."""

from __future__ import annotations


class TestBingham:
    def test_yield_threshold(self) -> None:
        from naviertwin.core.rheology.bingham import bingham_stress

        assert bingham_stress(gamma_dot=0.0, tau_y=2.0, mu_p=1.0) == 0
        # γ̇ > 0 → τ_y + μ_p γ̇
        assert bingham_stress(gamma_dot=3.0, tau_y=2.0, mu_p=1.0) == 5.0

    def test_apparent_viscosity_high_shear(self) -> None:
        from naviertwin.core.rheology.bingham import bingham_apparent_viscosity

        # high γ̇ → μ_app → μ_p
        v = bingham_apparent_viscosity(gamma_dot=1e6, tau_y=2.0, mu_p=1.0)
        assert abs(v - 1.0) < 1e-3
