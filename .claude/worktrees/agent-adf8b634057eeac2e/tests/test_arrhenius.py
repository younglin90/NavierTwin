"""Round 485 — Arrhenius."""

from __future__ import annotations


class TestArrhenius:
    def test_higher_T_higher_k(self) -> None:
        from naviertwin.core.reaction.arrhenius import arrhenius_k

        k1 = arrhenius_k(A=1e10, T=500, Ea=50000)
        k2 = arrhenius_k(A=1e10, T=1500, Ea=50000)
        assert k2 > k1

    def test_rate_first_order(self) -> None:
        from naviertwin.core.reaction.arrhenius import reaction_rate

        r = reaction_rate(k=2.0, concentrations=[3.0, 1.0], orders=[1.0, 0.0])
        assert abs(r - 6.0) < 1e-12
