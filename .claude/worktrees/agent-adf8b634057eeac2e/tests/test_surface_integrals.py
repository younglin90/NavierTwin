"""Round 608 — surface integrals: force, moment, coefficients."""

from __future__ import annotations

import numpy as np
import pytest


def _unit_square(z: float = 0.0):
    """Two triangles forming a unit square in the z=z plane, normal = +z."""
    p00 = [0.0, 0.0, z]
    p10 = [1.0, 0.0, z]
    p11 = [1.0, 1.0, z]
    p01 = [0.0, 1.0, z]
    tri1 = [p00, p10, p11]  # CCW from +z
    tri2 = [p00, p11, p01]
    return np.array([tri1, tri2], dtype=np.float64)


class TestNormalArea:
    def test_unit_square_area(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import (
            triangle_normal_area,
        )

        tris = _unit_square()
        n, A = triangle_normal_area(tris)
        assert A.shape == (2,)
        np.testing.assert_allclose(A.sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(n, [[0, 0, 1], [0, 0, 1]], atol=1e-12)

    def test_invalid_shape_raises(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import (
            triangle_normal_area,
        )

        with pytest.raises(ValueError, match="shape"):
            triangle_normal_area(np.zeros((5, 4, 3)))


class TestPressureForce:
    def test_uniform_pressure(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import pressure_force

        tris = _unit_square()
        P = np.array([1.0, 1.0])
        F = pressure_force(tris, P)
        # F = -∫ p n dA = -1 * 1 * (0,0,1) = (0, 0, -1)
        np.testing.assert_allclose(F, [0, 0, -1.0], atol=1e-12)

    def test_reference_pressure_subtraction(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import pressure_force

        tris = _unit_square()
        P = np.array([5.0, 5.0])
        F = pressure_force(tris, P, reference_pressure=5.0)
        np.testing.assert_allclose(F, [0, 0, 0], atol=1e-12)

    def test_pressure_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import pressure_force

        tris = _unit_square()
        with pytest.raises(ValueError, match="pressure"):
            pressure_force(tris, np.array([1.0]))


class TestViscousForce:
    def test_uniform_shear(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import viscous_force

        tris = _unit_square()
        # 모든 면에 +x 방향 0.5 단위 전단
        shear = np.tile([0.5, 0.0, 0.0], (2, 1))
        F = viscous_force(tris, shear)
        # 총면적 1 → F_x = 0.5 * 1 = 0.5
        np.testing.assert_allclose(F, [0.5, 0, 0], atol=1e-12)

    def test_shear_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import viscous_force

        tris = _unit_square()
        with pytest.raises(ValueError, match="shear_traction"):
            viscous_force(tris, np.zeros((1, 3)))


class TestTotalForce:
    def test_pressure_only(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import total_force

        tris = _unit_square()
        P = np.array([2.0, 2.0])
        F = total_force(tris, P)
        np.testing.assert_allclose(F, [0, 0, -2.0], atol=1e-12)

    def test_pressure_plus_shear(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import total_force

        tris = _unit_square()
        P = np.array([1.0, 1.0])
        shear = np.tile([0.3, 0.0, 0.0], (2, 1))
        F = total_force(tris, P, shear)
        np.testing.assert_allclose(F, [0.3, 0.0, -1.0], atol=1e-12)


class TestMoment:
    def test_zero_at_centroid(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import moment_about

        tris = _unit_square()
        P = np.array([1.0, 1.0])
        # square centroid = (0.5, 0.5, 0)
        M = moment_about(tris, P, center=np.array([0.5, 0.5, 0.0]))
        # 균일 압력 → 합력은 중심에 있으므로 모멘트 0이 아니지만 작음
        assert M.shape == (3,)

    def test_nonzero_offset_center(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import moment_about

        tris = _unit_square()
        P = np.array([1.0, 1.0])
        M_origin = moment_about(tris, P, center=np.array([0.0, 0.0, 0.0]))
        assert M_origin.shape == (3,)
        # F_z = -1, applied at centroid (0.5, 0.5, 0), about origin
        # M = r × F = (0.5, 0.5, 0) × (0, 0, -1) = (-0.5, 0.5, 0)
        np.testing.assert_allclose(M_origin, [-0.5, 0.5, 0.0], atol=1e-12)


class TestCoefficients:
    def test_force_coefficient_positive(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import force_coefficient

        cf = force_coefficient(100.0, rho=1.225, U_inf=10.0, A_ref=1.0)
        assert cf > 0
        # C = 100 / (0.5 * 1.225 * 100 * 1) ≈ 1.633
        np.testing.assert_allclose(cf, 100 / (0.5 * 1.225 * 100), atol=1e-6)

    def test_force_coefficient_invalid_raises(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import force_coefficient

        with pytest.raises(ValueError, match="> 0"):
            force_coefficient(10.0, rho=0.0, U_inf=10.0, A_ref=1.0)

    def test_moment_coefficient(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import (
            moment_coefficient,
        )

        cm = moment_coefficient(50.0, rho=1.225, U_inf=10.0, A_ref=1.0, L_ref=2.0)
        np.testing.assert_allclose(cm, 50 / (0.5 * 1.225 * 100 * 2), atol=1e-6)

    def test_moment_coefficient_invalid_raises(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import (
            moment_coefficient,
        )

        with pytest.raises(ValueError, match="positive"):
            moment_coefficient(1.0, rho=1.0, U_inf=1.0, A_ref=1.0, L_ref=0.0)


class TestLiftDragSplit:
    def test_x_aligned_flow_drag_only(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import lift_drag_split

        F = np.array([10.0, 0.0, 0.0])
        flow = np.array([1.0, 0.0, 0.0])
        lift, drag = lift_drag_split(F, flow)
        assert abs(drag - 10.0) < 1e-12
        assert abs(lift) < 1e-12

    def test_lift_only(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import lift_drag_split

        F = np.array([0.0, 5.0, 0.0])
        flow = np.array([1.0, 0.0, 0.0])
        lift, drag = lift_drag_split(F, flow)
        assert abs(drag) < 1e-12
        assert abs(lift - 5.0) < 1e-12

    def test_explicit_lift_direction(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import lift_drag_split

        F = np.array([3.0, 4.0, 0.0])
        flow = np.array([1.0, 0.0, 0.0])
        lift, drag = lift_drag_split(F, flow, lift_direction=np.array([0, 1, 0]))
        assert abs(drag - 3.0) < 1e-12
        assert abs(lift - 4.0) < 1e-12

    def test_zero_perpendicular(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import lift_drag_split

        F = np.array([5.0, 0.0, 0.0])
        flow = np.array([1.0, 0.0, 0.0])
        lift, drag = lift_drag_split(F, flow)
        assert lift == 0.0
        assert drag == 5.0

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.surface_integrals import lift_drag_split

        with pytest.raises(ValueError, match="shape"):
            lift_drag_split(np.zeros(2), np.zeros(3))
