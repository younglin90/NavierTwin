"""Round 553 — cyclone."""

from __future__ import annotations


class TestCyclone:
    def test_d50_positive(self) -> None:
        from naviertwin.core.applied.cyclone import lapple_d50

        d = lapple_d50(W=0.1, mu=1.8e-5, Ne=5, Vi=15, rho_p=2000, rho_g=1.2)
        assert d > 0

    def test_efficiency(self) -> None:
        from naviertwin.core.applied.cyclone import fraction_efficiency

        # at dp=d50 → η = 0.5
        assert abs(fraction_efficiency(dp=1.0, d50=1.0) - 0.5) < 1e-12
        # very large dp → η → 1
        assert fraction_efficiency(dp=100.0, d50=1.0) > 0.99
