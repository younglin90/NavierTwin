"""Round 626 — Reynolds stress anisotropy: Lumley triangle, barycentric."""

from __future__ import annotations

import numpy as np
import pytest


class TestAnisotropyTensor:
    def test_isotropic_zero(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import anisotropy_tensor

        # 등방: R = (2k/3) I
        R = (2.0 / 3.0) * np.eye(3)
        b = anisotropy_tensor(R)
        np.testing.assert_allclose(b, 0.0, atol=1e-12)

    def test_zero_k_returns_zero(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import anisotropy_tensor

        R = np.zeros((3, 3))
        b = anisotropy_tensor(R)
        np.testing.assert_array_equal(b, np.zeros((3, 3)))

    def test_traceless(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import anisotropy_tensor

        rng = np.random.default_rng(0)
        R = rng.standard_normal((3, 3))
        R = R + R.T  # 대칭
        b = anisotropy_tensor(R)
        np.testing.assert_allclose(np.trace(b), 0.0, atol=1e-12)

    def test_invalid_shape_raises(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import anisotropy_tensor

        with pytest.raises(ValueError, match="3, 3"):
            anisotropy_tensor(np.eye(2))


class TestInvariants:
    def test_isotropic_zero(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import invariants_II_III

        II, III = invariants_II_III(np.zeros((3, 3)))
        assert II == 0.0
        assert III == 0.0

    def test_diagonal(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import invariants_II_III

        # b = diag(1/3, -1/6, -1/6) → trace = 0
        b = np.diag([1 / 3, -1 / 6, -1 / 6])
        II, III = invariants_II_III(b)
        # II = trace(b^2) = 1/9 + 1/36 + 1/36 = 4/36 + 1/36 + 1/36 = 6/36 = 1/6
        np.testing.assert_allclose(II, 1 / 6, atol=1e-12)

    def test_invalid_shape(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import invariants_II_III

        with pytest.raises(ValueError, match="3, 3"):
            invariants_II_III(np.eye(2))


class TestLumleyEtaXi:
    def test_isotropic_origin(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import lumley_eta_xi

        eta, xi = lumley_eta_xi(np.zeros((3, 3)))
        assert eta == 0.0
        assert xi == 0.0

    def test_axisymmetric_expansion_xi_pos(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import lumley_eta_xi

        # b = diag(1/3, -1/6, -1/6) → III > 0 → ξ > 0 (axisymmetric expansion)
        b = np.diag([1 / 3, -1 / 6, -1 / 6])
        eta, xi = lumley_eta_xi(b)
        assert eta > 0
        assert xi > 0

    def test_axisymmetric_contraction_xi_neg(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import lumley_eta_xi

        # b = diag(-1/3, 1/6, 1/6) → III < 0 → ξ < 0
        b = np.diag([-1 / 3, 1 / 6, 1 / 6])
        eta, xi = lumley_eta_xi(b)
        assert eta > 0
        assert xi < 0


class TestRealizability:
    def test_isotropic_realizable(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import is_realizable

        assert is_realizable(np.zeros((3, 3))) is True

    def test_extreme_unrealizable(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import is_realizable

        # 매우 큰 비등방
        b = np.eye(3) * 5.0
        assert is_realizable(b) is False


class TestTurbulenceState:
    def test_isotropic(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import turbulence_state

        assert turbulence_state(np.zeros((3, 3))) == "isotropic"

    def test_axisymmetric_expansion(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import turbulence_state

        b = np.diag([1 / 3, -1 / 6, -1 / 6])
        # ξ > 0
        state = turbulence_state(b)
        assert state in ("axisymmetric_expansion", "1_component", "2_component")


class TestEigenvalues:
    def test_diagonal_returns_sorted(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import (
            reynolds_stress_eigenvalues,
        )

        R = np.diag([3.0, 1.0, 2.0])
        eigs = reynolds_stress_eigenvalues(R)
        np.testing.assert_allclose(eigs, [1.0, 2.0, 3.0])

    def test_invalid_shape(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import (
            reynolds_stress_eigenvalues,
        )

        with pytest.raises(ValueError, match="3, 3"):
            reynolds_stress_eigenvalues(np.eye(2))


class TestBarycentric:
    def test_isotropic_3c(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import barycentric_coordinates

        # b = 0 → eigs all 0 → C_3c = 1
        c1, c2, c3 = barycentric_coordinates(np.zeros((3, 3)))
        np.testing.assert_allclose(c3, 1.0, atol=1e-12)
        np.testing.assert_allclose(c1, 0.0, atol=1e-12)

    def test_sum_unity(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import barycentric_coordinates

        rng = np.random.default_rng(2)
        b = rng.standard_normal((3, 3))
        b = (b + b.T) / 2
        # traceless
        b = b - np.trace(b) / 3 * np.eye(3)
        c1, c2, c3 = barycentric_coordinates(b)
        np.testing.assert_allclose(c1 + c2 + c3, 1.0, atol=1e-12)

    def test_invalid_shape(self) -> None:
        from naviertwin.core.flow_analysis.anisotropy import barycentric_coordinates

        with pytest.raises(ValueError, match="3, 3"):
            barycentric_coordinates(np.eye(2))
