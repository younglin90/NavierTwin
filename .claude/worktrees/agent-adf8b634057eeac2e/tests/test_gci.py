"""Round 502 — GCI."""

from __future__ import annotations


class TestGCI:
    def test_basic(self) -> None:
        from naviertwin.core.verification.gci import gci

        # 2x refine, 2nd-order, eps=0.01 → GCI = 1.25 * 0.01 / 3 ≈ 0.00417
        v = gci(eps=0.01, r=2.0, p=2.0)
        assert abs(v - 0.00417) < 1e-4

    def test_observed_order(self) -> None:
        from naviertwin.core.verification.gci import observed_order

        # f exact = 0; f_h = h^2 → for r=2 grids: f1=h², f2=4h², f3=16h²
        # p = ln((16-4)/(4-1))/ln(2) = ln(4)/ln(2) = 2
        p = observed_order(f1=1.0, f2=4.0, f3=16.0, r=2.0)
        assert abs(p - 2.0) < 1e-9
