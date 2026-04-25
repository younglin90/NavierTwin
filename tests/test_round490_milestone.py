"""Round 490 — W category milestone: rheology + reaction (R481-R489) e2e."""

from __future__ import annotations


class TestMilestoneW:
    def test_imports(self) -> None:
        from naviertwin.core.reaction import (  # noqa: F401
            adiabatic_flame,
            arrhenius,
            chemkin_parser,
            mixture_fraction,
            soret_dufour,
        )
        from naviertwin.core.rheology import (  # noqa: F401
            bingham,
            carreau_yasuda,
            cross_model,
            power_law,
        )

    def test_power_law_arrhenius_e2e(self) -> None:
        from naviertwin.core.reaction.arrhenius import arrhenius_k, reaction_rate
        from naviertwin.core.rheology.power_law import apparent_viscosity

        # power-law fluid in reactor; high T → fast rate
        mu = apparent_viscosity(gamma_dot=10.0, K=1.0, n=0.5)
        k = arrhenius_k(A=1e6, T=800, Ea=80000)
        r = reaction_rate(k=k, concentrations=[1.0, 0.5], orders=[1.0, 1.0])
        assert mu > 0
        assert r > 0
