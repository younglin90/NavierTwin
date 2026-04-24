"""Round 105 — 공력 계수."""

from __future__ import annotations

import numpy as np
import pytest


class TestAeroForces:
    def test_cp(self) -> None:
        from naviertwin.core.analysis.aero_forces import pressure_coefficient

        cp = pressure_coefficient(
            np.array([100.0, 150.0, 200.0]),
            p_ref=100.0, rho=1.0, U_ref=10.0,
        )
        assert np.allclose(cp, [0.0, 1.0, 2.0])

    def test_cp_zero_dyn_pressure(self) -> None:
        from naviertwin.core.analysis.aero_forces import pressure_coefficient

        with pytest.raises(ValueError):
            pressure_coefficient(np.zeros(3), 0.0, 0.0, 0.0)

    def test_uniform_pressure_on_sphere(self) -> None:
        """구면 균일 압력 → 합력 = 0."""
        from naviertwin.core.analysis.aero_forces import surface_force

        # 간단히 정반대 법선 쌍
        normals = np.array([
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        ])
        areas = np.ones(4)
        p = np.ones(4) * 100.0
        F = surface_force(p, normals, areas)
        assert np.allclose(F, 0.0, atol=1e-12)

    def test_drag_lift(self) -> None:
        from naviertwin.core.analysis.aero_forces import drag_lift_coefficients

        F = np.array([10.0, 5.0, 0.0])
        out = drag_lift_coefficients(F, rho=1.0, U_ref=10.0, area_ref=1.0)
        assert out["Cd"] == pytest.approx(10.0 / 50.0)
        assert out["Cl"] == pytest.approx(5.0 / 50.0)
