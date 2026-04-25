"""Round 459 — cooling tower."""

from __future__ import annotations


class TestCT:
    def test_eps(self) -> None:
        from naviertwin.core.applied.cooling_tower import tower_effectiveness

        eps = tower_effectiveness(T_h_in=40, T_h_out=30, T_wb=25)
        assert abs(eps - 10 / 15) < 1e-12

    def test_ntu_monotone(self) -> None:
        from naviertwin.core.applied.cooling_tower import NTU_from_effectiveness

        n1 = NTU_from_effectiveness(eps=0.5)
        n2 = NTU_from_effectiveness(eps=0.9)
        assert n2 > n1
