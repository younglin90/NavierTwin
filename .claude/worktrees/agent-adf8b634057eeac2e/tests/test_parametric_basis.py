"""Round 644 — parametric POD: alignment + Grassmann interpolation."""

from __future__ import annotations

import numpy as np
import pytest


class TestAlignBases:
    def test_identical_unchanged(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            align_bases,
        )

        rng = np.random.default_rng(0)
        Q, _ = np.linalg.qr(rng.standard_normal((20, 4)))
        aligned = align_bases([Q, Q, Q])
        for a in aligned:
            np.testing.assert_allclose(a, Q, atol=1e-10)

    def test_sign_flip_aligned(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            align_bases,
        )

        rng = np.random.default_rng(1)
        Q, _ = np.linalg.qr(rng.standard_normal((20, 3)))
        Q_flip = Q.copy()
        Q_flip[:, 0] = -Q_flip[:, 0]
        aligned = align_bases([Q, Q_flip])
        # 정렬 후 두 기저가 가까워짐
        diff_before = np.linalg.norm(Q - Q_flip)
        diff_after = np.linalg.norm(aligned[0] - aligned[1])
        assert diff_after < diff_before

    def test_empty_returns_empty(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            align_bases,
        )

        assert align_bases([]) == []

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            align_bases,
        )

        with pytest.raises(ValueError, match="same shape"):
            align_bases([np.zeros((10, 3)), np.zeros((10, 4))])

    def test_reference_shape_mismatch(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            align_bases,
        )

        with pytest.raises(ValueError, match="reference"):
            align_bases(
                [np.zeros((10, 3))],
                reference=np.zeros((10, 4)),
            )


class TestGrassmannLogExp:
    def test_log_exp_identity(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            grassmann_exp,
            grassmann_log,
        )

        rng = np.random.default_rng(2)
        Q0, _ = np.linalg.qr(rng.standard_normal((20, 4)))
        # 작은 perturbation으로 가까운 부분공간 생성
        delta = 0.05 * rng.standard_normal((20, 4))
        Q1, _ = np.linalg.qr(Q0 + delta)
        Gamma = grassmann_log(Q0, Q1)
        Q1_recovered = grassmann_exp(Q0, Gamma)
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            subspace_angles,
        )

        angles = subspace_angles(Q1_recovered, Q1)
        assert np.max(angles) < 0.05

    def test_zero_log_for_same(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            grassmann_log,
        )

        rng = np.random.default_rng(3)
        Q, _ = np.linalg.qr(rng.standard_normal((20, 4)))
        Gamma = grassmann_log(Q, Q)
        assert np.linalg.norm(Gamma) < 1e-8

    def test_exp_with_zero_returns_Y0(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            grassmann_exp,
        )

        rng = np.random.default_rng(4)
        Q, _ = np.linalg.qr(rng.standard_normal((20, 4)))
        result = grassmann_exp(Q, np.zeros((20, 4)))
        # Subspace 동일
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            subspace_angles,
        )

        angles = subspace_angles(result, Q)
        assert np.max(angles) < 1e-6

    def test_log_shape_mismatch(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            grassmann_log,
        )

        with pytest.raises(ValueError, match="shape"):
            grassmann_log(np.zeros((10, 3)), np.zeros((10, 4)))

    def test_exp_shape_mismatch(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            grassmann_exp,
        )

        with pytest.raises(ValueError, match="shape"):
            grassmann_exp(np.zeros((10, 3)), np.zeros((10, 4)))


class TestLinearInterpolate:
    def test_endpoint_returns_known(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            linear_interpolate_bases,
        )

        rng = np.random.default_rng(5)
        Q1, _ = np.linalg.qr(rng.standard_normal((20, 3)))
        Q2, _ = np.linalg.qr(rng.standard_normal((20, 3)))
        params = np.array([0.0, 1.0])
        result = linear_interpolate_bases([Q1, Q2], params, target=0.0)
        assert result.shape == Q1.shape

    def test_clamp_outside_range(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            linear_interpolate_bases,
        )

        rng = np.random.default_rng(6)
        Q1, _ = np.linalg.qr(rng.standard_normal((15, 2)))
        Q2, _ = np.linalg.qr(rng.standard_normal((15, 2)))
        params = np.array([0.0, 1.0])
        # target < min
        result = linear_interpolate_bases([Q1, Q2], params, target=-1.0)
        assert result.shape == Q1.shape

    def test_midpoint_returns_valid_basis(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            linear_interpolate_bases,
        )

        rng = np.random.default_rng(7)
        bases = [np.linalg.qr(rng.standard_normal((20, 3)))[0] for _ in range(3)]
        params = np.array([0.0, 1.0, 2.0])
        result = linear_interpolate_bases(bases, params, target=0.5)
        # 직교 정규
        np.testing.assert_allclose(result.T @ result, np.eye(3), atol=1e-8)

    def test_length_mismatch_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            linear_interpolate_bases,
        )

        with pytest.raises(ValueError, match="length"):
            linear_interpolate_bases(
                [np.zeros((10, 2))], np.array([0.0, 1.0]), target=0.5,
            )

    def test_too_few_bases(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            linear_interpolate_bases,
        )

        with pytest.raises(ValueError, match="2 bases"):
            linear_interpolate_bases(
                [np.zeros((10, 2))], np.array([0.0]), target=0.5,
            )

    def test_non_increasing_params(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            linear_interpolate_bases,
        )

        with pytest.raises(ValueError, match="increasing"):
            linear_interpolate_bases(
                [np.zeros((10, 2))] * 3,
                np.array([0.0, 0.5, 0.3]),
                target=0.4,
            )


class TestBasisDistanceCurve:
    def test_zero_for_identical(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            basis_distance_curve,
        )

        rng = np.random.default_rng(8)
        Q, _ = np.linalg.qr(rng.standard_normal((20, 3)))
        d = basis_distance_curve([Q, Q, Q])
        np.testing.assert_allclose(d, 0.0, atol=1e-8)

    def test_increasing_distance(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            basis_distance_curve,
        )

        rng = np.random.default_rng(9)
        bases = [np.linalg.qr(rng.standard_normal((20, 3)))[0] for _ in range(4)]
        d = basis_distance_curve(bases)
        assert d.shape == (3,)
        assert np.all(d >= 0)

    def test_empty_or_single(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            basis_distance_curve,
        )

        assert basis_distance_curve([]).shape == (0,)
        assert basis_distance_curve(
            [np.zeros((10, 3))],
        ).shape == (0,)

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.parametric_basis import (
            basis_distance_curve,
        )

        with pytest.raises(ValueError, match="shape"):
            basis_distance_curve(
                [np.zeros((10, 3)), np.zeros((10, 4))],
            )
