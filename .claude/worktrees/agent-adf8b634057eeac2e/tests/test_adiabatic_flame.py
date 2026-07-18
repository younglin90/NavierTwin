"""Round 486 — adiabatic flame T."""

from __future__ import annotations


class TestAFT:
    def test_methane_order(self) -> None:
        from naviertwin.core.reaction.adiabatic_flame import T_adiabatic

        # CH4 combustion ΔH_r ≈ 802 kJ/mol; products ≈ 10 mol; cp ≈ 40
        T = T_adiabatic(T_in=298, dHr=802000, n_fuel=1.0, n_products=10.0,
                          cp_avg=40.0)
        # Crude estimate ~ 2300 K (real CH4 AFT ≈ 2200 K)
        assert 2000 < T < 2500
