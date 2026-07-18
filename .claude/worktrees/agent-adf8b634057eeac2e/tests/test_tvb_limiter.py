"""Round 398 — TVB limiter."""

from __future__ import annotations


class TestTVB:
    def test_within_threshold(self) -> None:
        from naviertwin.core.solvers.tvb_limiter import tvb_minmod

        # |a|=0.05 ≤ M h² = 10 * 0.01 = 0.1, so a returned unchanged
        assert tvb_minmod(0.05, 1.0, 2.0, M=10.0, h=0.1) == 0.05

    def test_minmod_active(self) -> None:
        from naviertwin.core.solvers.tvb_limiter import tvb_minmod

        # all positive, take min
        v = tvb_minmod(1.0, 2.0, 3.0, M=0.0, h=0.1)
        assert v == 1.0

    def test_opposite_signs_zero(self) -> None:
        from naviertwin.core.solvers.tvb_limiter import tvb_minmod

        v = tvb_minmod(1.0, -2.0, 3.0, M=0.0, h=0.1)
        assert v == 0.0
