"""Round 453 — pipe flow."""

from __future__ import annotations


class TestPipe:
    def test_dp(self) -> None:
        from naviertwin.core.applied.pipe_flow import pressure_drop

        # f=0.02, L/D=100, ½ρU²=500 → Δp = 0.02*100*500 = 1000
        assert pressure_drop(L=10, D=0.1, rho=1000, U=1.0, f=0.02) == 1000.0

    def test_colebrook_smooth(self) -> None:
        from naviertwin.core.applied.pipe_flow import friction_colebrook

        # smooth pipe at Re=1e5: f ≈ 0.018
        f = friction_colebrook(Re=1e5, eps_over_D=0.0)
        assert 0.015 < f < 0.025

    def test_colebrook_rough(self) -> None:
        from naviertwin.core.applied.pipe_flow import friction_colebrook

        # rough pipe → higher f
        f_smooth = friction_colebrook(Re=1e5, eps_over_D=0.0)
        f_rough = friction_colebrook(Re=1e5, eps_over_D=0.01)
        assert f_rough > f_smooth
