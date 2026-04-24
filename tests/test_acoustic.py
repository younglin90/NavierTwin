"""Round 39 — acoustic modes + Strouhal/Womersley."""

from __future__ import annotations

import numpy as np
import pytest


class TestDuctModes:
    def test_dirichlet_fundamental(self) -> None:
        from naviertwin.core.flow_analysis.acoustic import duct_modes_dirichlet

        freqs, modes = duct_modes_dirichlet(L=1.0, c=340.0, n_modes=3, n_points=50)
        # f_1 = c / (2L) = 170
        assert abs(freqs[0] - 170.0) < 0.1
        # f_2 = 340, f_3 = 510
        assert abs(freqs[1] - 340.0) < 0.1
        assert abs(freqs[2] - 510.0) < 0.1
        # 경계 0
        assert modes[0, 0] == 0.0
        assert abs(modes[-1, 0]) < 1e-10

    def test_neumann_zeroth_mode(self) -> None:
        from naviertwin.core.flow_analysis.acoustic import duct_modes_neumann

        freqs, modes = duct_modes_neumann(L=1.0, c=340, n_modes=4, n_points=30)
        assert freqs[0] == 0.0  # DC
        # 첫 번째 비트리비얼 = c/(2L)
        assert abs(freqs[1] - 170.0) < 0.1


class TestStrouhalWomersley:
    def test_strouhal(self) -> None:
        from naviertwin.core.flow_analysis.acoustic import strouhal

        # vortex shedding cylinder: St ≈ 0.2 at Re ≈ 10⁴
        st = strouhal(f=2.0, L=0.1, U=1.0)
        assert st == pytest.approx(0.2)

    def test_strouhal_invalid(self) -> None:
        from naviertwin.core.flow_analysis.acoustic import strouhal

        with pytest.raises(ValueError):
            strouhal(f=1.0, L=-1.0, U=1.0)

    def test_womersley(self) -> None:
        from naviertwin.core.flow_analysis.acoustic import womersley

        # α = R·sqrt(ω/ν) — 혈관 직경/ν 스케일 따라 다양
        alpha = womersley(omega=2 * np.pi * 1.2, R=0.01, nu=3.5e-6)
        assert 2.0 < alpha < 50.0
