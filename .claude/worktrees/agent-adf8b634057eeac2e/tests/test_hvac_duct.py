"""Round 457 — HVAC duct."""

from __future__ import annotations


class TestHVAC:
    def test_loss(self) -> None:
        from naviertwin.core.applied.hvac_duct import total_pressure_loss

        # f L/D = 0.02 * 10/0.3 ≈ 0.667; K=2; ½ρU² = 0.5*1.2*25=15
        # → (0.667 + 2)*15 ≈ 40.0
        dp = total_pressure_loss(L=10, D=0.3, rho=1.2, U=5, f=0.02, K_total=2.0)
        assert 39 < dp < 42

    def test_velocity(self) -> None:
        from naviertwin.core.applied.hvac_duct import duct_velocity

        v = duct_velocity(mdot=1.2, rho=1.2, A=0.1)
        assert v == 10.0
