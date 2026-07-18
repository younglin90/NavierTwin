"""Round 452 — heat exchanger ε-NTU."""

from __future__ import annotations


class TestHX:
    def test_eps_in_range(self) -> None:
        from naviertwin.core.applied.heat_exchanger import effectiveness

        eps = effectiveness(NTU=2.0, Cr=0.5, flow="counterflow")
        assert 0 < eps < 1

    def test_higher_NTU_higher_eps(self) -> None:
        from naviertwin.core.applied.heat_exchanger import effectiveness

        e1 = effectiveness(NTU=1.0, Cr=0.5)
        e2 = effectiveness(NTU=5.0, Cr=0.5)
        assert e2 > e1

    def test_q_formula(self) -> None:
        from naviertwin.core.applied.heat_exchanger import heat_transfer_rate

        q = heat_transfer_rate(eps=0.5, C_min=2.0, T_h_in=100, T_c_in=20)
        assert q == 0.5 * 2 * 80
