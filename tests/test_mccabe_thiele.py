"""Round 554 — McCabe-Thiele."""

from __future__ import annotations


class TestMT:
    def test_eq(self) -> None:
        from naviertwin.core.applied.mccabe_thiele import equilibrium_y

        # α=1 → y=x
        assert equilibrium_y(x=0.5, alpha=1.0) == 0.5
        # α=4, x=0.5 → 2/(1+1.5) = 0.8
        assert abs(equilibrium_y(x=0.5, alpha=4.0) - 0.8) < 1e-9

    def test_op_line_inverse(self) -> None:
        from naviertwin.core.applied.mccabe_thiele import op_line_x

        # R=1, x_D=1; y=(1/2) x + 1/2.  At x=0 → y=0.5; invert: 0.5 → 0
        assert abs(op_line_x(y=0.5, R=1.0, x_D=1.0)) < 1e-12
