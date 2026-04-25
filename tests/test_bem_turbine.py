"""Round 451 — BEM turbine."""

from __future__ import annotations


class TestBEM:
    def test_betz(self) -> None:
        from naviertwin.core.applied.bem_turbine import betz_limit

        assert abs(betz_limit() - 16.0 / 27.0) < 1e-12

    def test_cp_under_betz(self) -> None:
        from naviertwin.core.applied.bem_turbine import betz_limit, cp_estimate

        cp = cp_estimate(tip_speed_ratio=7.0, cl=1.0, cd=0.05)
        assert 0 < cp < betz_limit()
