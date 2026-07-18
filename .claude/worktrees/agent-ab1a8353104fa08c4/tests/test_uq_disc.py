"""Round 508 — discretization UQ."""

from __future__ import annotations


class TestUQDisc:
    def test_rss(self) -> None:
        from naviertwin.core.verification.uq_disc import combined_disc_uncertainty

        # √(3² + 4²) = 5
        assert abs(combined_disc_uncertainty([3.0, 4.0]) - 5.0) < 1e-12
