"""Round 488 — Soret/Dufour."""

from __future__ import annotations


class TestSoret:
    def test_no_gradT(self) -> None:
        from naviertwin.core.reaction.soret_dufour import soret_flux

        # gradT=0 → Soret term vanishes; only Fickian
        j = soret_flux(rho=1, D=1e-5, D_T=1e-7, Y=0.5, gradY=2.0, gradT=0.0, T=300)
        assert abs(j - (-2e-5)) < 1e-12

    def test_dufour(self) -> None:
        from naviertwin.core.reaction.soret_dufour import dufour_heat

        assert dufour_heat(k_dufour=2.0, gradY=3.0) == -6.0
