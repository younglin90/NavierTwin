"""Round 489 — mixture fraction Bilger."""

from __future__ import annotations


class TestZ:
    def test_pure_oxidizer(self) -> None:
        from naviertwin.core.reaction.mixture_fraction import bilger_Z

        assert bilger_Z(beta=0.0, beta_fuel=1.0, beta_ox=0.0) == 0.0

    def test_pure_fuel(self) -> None:
        from naviertwin.core.reaction.mixture_fraction import bilger_Z

        assert bilger_Z(beta=1.0, beta_fuel=1.0, beta_ox=0.0) == 1.0

    def test_midpoint(self) -> None:
        from naviertwin.core.reaction.mixture_fraction import bilger_Z

        assert bilger_Z(beta=0.5, beta_fuel=1.0, beta_ox=0.0) == 0.5
