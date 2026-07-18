"""Round 460 — FINAL milestone T: applied scenarios e2e."""

from __future__ import annotations


class TestMilestoneT:
    def test_imports(self) -> None:
        from naviertwin.core.applied import (  # noqa: F401
            battery_thermal,
            bem_turbine,
            compressor_map,
            cooling_tower,
            heat_exchanger,
            hvac_duct,
            pipe_flow,
            solar_pv,
            turbo_match,
        )

    def test_pipe_heat_ex_e2e(self) -> None:
        """Pipe Δp + heat exchanger ε-NTU."""
        from naviertwin.core.applied.heat_exchanger import (
            effectiveness,
            heat_transfer_rate,
        )
        from naviertwin.core.applied.pipe_flow import (
            friction_colebrook,
            pressure_drop,
        )

        f = friction_colebrook(Re=5e4, eps_over_D=0.001)
        dp = pressure_drop(L=20, D=0.05, rho=1000, U=2.0, f=f)
        assert dp > 0
        eps = effectiveness(NTU=2.0, Cr=0.4, flow="counterflow")
        q = heat_transfer_rate(eps=eps, C_min=5000, T_h_in=80, T_c_in=20)
        assert q > 0

    def test_full_block_smoke(self) -> None:
        from naviertwin.core.applied.bem_turbine import cp_estimate
        from naviertwin.core.applied.solar_pv import iv_curve, mppt

        assert 0 < cp_estimate(tip_speed_ratio=8.0) < 0.6
        V, cur = iv_curve()
        Vm, Im, Pm = mppt(V, cur)
        assert Pm > 0
