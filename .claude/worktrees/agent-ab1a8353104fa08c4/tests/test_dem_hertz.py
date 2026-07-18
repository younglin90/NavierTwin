"""Round 478 — DEM Hertz."""

from __future__ import annotations


class TestHertz:
    def test_no_overlap(self) -> None:
        from naviertwin.core.meshless.dem_hertz import hertz_force

        assert hertz_force(delta=-0.001, E_star=1e9, R_star=0.01) == 0
        assert hertz_force(delta=0.0, E_star=1e9, R_star=0.01) == 0

    def test_force_scaling(self) -> None:
        from naviertwin.core.meshless.dem_hertz import hertz_force

        f1 = hertz_force(delta=1e-3, E_star=1e9, R_star=0.01)
        f2 = hertz_force(delta=4e-3, E_star=1e9, R_star=0.01)
        # δ^1.5: f2/f1 = 4^1.5 = 8
        assert abs(f2 / f1 - 8.0) < 1e-6

    def test_combined_E_R(self) -> None:
        from naviertwin.core.meshless.dem_hertz import E_star, R_star

        assert E_star(E1=2e11, nu1=0.3, E2=2e11, nu2=0.3) > 0
        assert R_star(R1=0.01, R2=0.01) == 0.005
