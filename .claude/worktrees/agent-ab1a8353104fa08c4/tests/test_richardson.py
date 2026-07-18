"""Round 503 — Richardson extrapolation."""

from __future__ import annotations


class TestRichardson:
    def test_quadratic(self) -> None:
        from naviertwin.core.verification.richardson import richardson

        # f(h)=1+h²; f(1)=2, f(0.5)=1.25; r=2, p=2 → R = 1
        v = richardson(f_fine=1.25, f_coarse=2.0, r=2.0, p=2.0)
        assert abs(v - 1.0) < 1e-9
