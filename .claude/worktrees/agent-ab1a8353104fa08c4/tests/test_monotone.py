"""Round 507 — monotone convergence."""

from __future__ import annotations


class TestMonotone:
    def test_decreasing(self) -> None:
        from naviertwin.core.verification.monotone import is_monotone_decreasing

        assert is_monotone_decreasing([1.0, 0.5, 0.1])

    def test_not_decreasing(self) -> None:
        from naviertwin.core.verification.monotone import is_monotone_decreasing

        assert not is_monotone_decreasing([1.0, 2.0, 0.5])

    def test_ratio(self) -> None:
        from naviertwin.core.verification.monotone import convergence_ratio

        r = convergence_ratio([1.0, 0.25, 0.0625])
        assert r == [0.25, 0.25]
