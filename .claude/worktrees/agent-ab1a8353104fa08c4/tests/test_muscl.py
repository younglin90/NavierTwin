"""Round 392 — MUSCL limiters."""

from __future__ import annotations


class TestMUSCL:
    def test_minmod(self) -> None:
        from naviertwin.core.solvers.muscl import minmod

        assert minmod(1.0, 2.0) == 1.0
        assert minmod(-1.0, -2.0) == -1.0
        assert minmod(1.0, -2.0) == 0.0  # opposite signs

    def test_van_leer(self) -> None:
        from naviertwin.core.solvers.muscl import van_leer

        assert abs(van_leer(1.0, 1.0) - 1.0) < 1e-12
        assert van_leer(1.0, -2.0) == 0.0

    def test_superbee(self) -> None:
        from naviertwin.core.solvers.muscl import superbee

        # superbee at equal slopes: r=1 → φ=1
        assert superbee(1.0, 1.0) > 0
        assert superbee(-1.0, 2.0) == 0.0
