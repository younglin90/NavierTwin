"""Round 612 — plane flux integrals (mass, momentum, scalar, KE)."""

from __future__ import annotations

import numpy as np
import pytest


def _unit_square():
    """Two triangles forming unit square in z=0 plane, normal +z."""
    p00, p10, p11, p01 = (
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    )
    return np.array([[p00, p10, p11], [p00, p11, p01]], dtype=np.float64)


class TestMassFlux:
    def test_unit_velocity_unit_density(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import mass_flux

        tris = _unit_square()
        u = np.array([[0, 0, 1.0], [0, 0, 1.0]])
        m = mass_flux(tris, u, density=1.0)
        np.testing.assert_allclose(m, 1.0, atol=1e-12)

    def test_density_array(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import mass_flux

        tris = _unit_square()
        u = np.array([[0, 0, 2.0], [0, 0, 2.0]])
        rho = np.array([0.5, 1.5])
        m = mass_flux(tris, u, density=rho)
        # 각 삼각형 면적 0.5 → m = 0.5 * 2 * 0.5 + 0.5 * 2 * 1.5 = 0.5 + 1.5 = 2
        np.testing.assert_allclose(m, 2.0, atol=1e-12)

    def test_zero_normal_velocity(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import mass_flux

        tris = _unit_square()
        # 평면 평행 흐름
        u = np.array([[1.0, 0, 0], [1.0, 0, 0]])
        m = mass_flux(tris, u)
        np.testing.assert_allclose(m, 0.0, atol=1e-12)

    def test_velocity_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import mass_flux

        tris = _unit_square()
        with pytest.raises(ValueError, match="velocity"):
            mass_flux(tris, np.zeros((1, 3)))

    def test_density_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import mass_flux

        tris = _unit_square()
        u = np.zeros((2, 3))
        with pytest.raises(ValueError, match="density"):
            mass_flux(tris, u, density=np.array([1.0, 1.0, 1.0]))


class TestVolumetricFlow:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import volumetric_flow_rate

        tris = _unit_square()
        u = np.array([[0, 0, 3.0], [0, 0, 3.0]])
        Q = volumetric_flow_rate(tris, u)
        np.testing.assert_allclose(Q, 3.0, atol=1e-12)


class TestMomentumFlux:
    def test_z_velocity(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import momentum_flux

        tris = _unit_square()
        u = np.array([[0, 0, 2.0], [0, 0, 2.0]])
        M = momentum_flux(tris, u, density=1.0)
        # ∫ ρ u (u·n) dA = ρ |u|² A 만큼 = 1 * 2 * 1 * 2 = 4
        np.testing.assert_allclose(M[2], 4.0, atol=1e-12)

    def test_velocity_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import momentum_flux

        tris = _unit_square()
        with pytest.raises(ValueError, match="velocity"):
            momentum_flux(tris, np.zeros((1, 3)))


class TestScalarFlux:
    def test_temperature_flux(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import scalar_flux

        tris = _unit_square()
        u = np.array([[0, 0, 1.0], [0, 0, 1.0]])
        T = np.array([300.0, 300.0])
        flux = scalar_flux(tris, u, T)
        np.testing.assert_allclose(flux, 300.0, atol=1e-12)

    def test_scalar_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import scalar_flux

        tris = _unit_square()
        u = np.zeros((2, 3))
        with pytest.raises(ValueError, match="scalar"):
            scalar_flux(tris, u, np.array([1.0]))


class TestKineticEnergyFlux:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import kinetic_energy_flux

        tris = _unit_square()
        u = np.array([[0, 0, 2.0], [0, 0, 2.0]])
        ke = kinetic_energy_flux(tris, u, density=1.0)
        # ½|u|² (u·n) A = 0.5 * 4 * 2 * 1 = 4
        np.testing.assert_allclose(ke, 4.0, atol=1e-12)


class TestAreaAverage:
    def test_scalar_field(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import area_average

        tris = _unit_square()
        T = np.array([300.0, 350.0])
        avg = area_average(tris, T)
        # 면적 가중 평균 (각 면 0.5)
        np.testing.assert_allclose(avg, 325.0, atol=1e-12)

    def test_vector_field(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import area_average

        tris = _unit_square()
        u = np.array([[1.0, 0, 0], [3.0, 0, 0]])
        avg = area_average(tris, u)
        np.testing.assert_allclose(avg, [2.0, 0, 0], atol=1e-12)

    def test_zero_area(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import area_average

        # 퇴화된 삼각형
        tris = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.float64)
        T = np.array([300.0])
        avg = area_average(tris, T)
        assert avg == 0.0

    def test_field_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import area_average

        tris = _unit_square()
        with pytest.raises(ValueError, match="field"):
            area_average(tris, np.array([1.0, 2.0, 3.0]))


class TestMassWeightedAvg:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import mass_weighted_average

        tris = _unit_square()
        u = np.array([[0, 0, 1.0], [0, 0, 2.0]])
        T = np.array([300.0, 400.0])
        # ṁ_i = 0.5 * 1 = 0.5; 0.5 * 2 = 1.0 → 가중치 비율 1:2
        # 평균 = (0.5*300 + 1*400)/(0.5+1) = (150+400)/1.5 = 366.67
        avg = mass_weighted_average(tris, u, T)
        np.testing.assert_allclose(avg, 550 / 1.5, atol=1e-12)

    def test_zero_mass_flux(self) -> None:
        from naviertwin.core.flow_analysis.plane_flux import mass_weighted_average

        tris = _unit_square()
        u = np.zeros((2, 3))
        T = np.array([300.0, 400.0])
        avg = mass_weighted_average(tris, u, T)
        assert avg == 0.0
