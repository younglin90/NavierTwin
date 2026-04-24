"""Round 395 — low-Mach preconditioning."""

from __future__ import annotations


class TestLowMach:
    def test_factor_clip(self) -> None:
        from naviertwin.core.solvers.low_mach import precond_factor

        assert precond_factor(0.5, M_cut=0.3) == 1.0  # clipped
        assert abs(precond_factor(0.03, M_cut=0.3) - 0.1) < 1e-12

    def test_speed_reduces(self) -> None:
        from naviertwin.core.solvers.low_mach import precond_speed

        c0 = 340.0
        c_p = precond_speed(M_local=0.01, M_cut=0.3, c=c0)
        assert c_p < c0
        assert c_p > 0

    def test_eigvals(self) -> None:
        from naviertwin.core.solvers.low_mach import precond_eigvals

        ev = precond_eigvals(u=10.0, c=340.0, M_local=0.05, M_cut=0.3)
        assert ev.shape == (3,)
