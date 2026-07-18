"""Round 633 — POD mode tracking + subspace angles + drift detection."""

from __future__ import annotations

import numpy as np
import pytest


class TestSubspaceAngles:
    def test_identical_zero_angles(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            subspace_angles,
        )

        A = np.eye(5)[:, :3]
        angles = subspace_angles(A, A)
        np.testing.assert_allclose(angles, 0.0, atol=1e-10)

    def test_orthogonal_pi_half(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            subspace_angles,
        )

        # 직교 부분공간: 각도 = π/2
        A = np.eye(5)[:, :2]
        B = np.eye(5)[:, 2:4]
        angles = subspace_angles(A, B)
        np.testing.assert_allclose(angles, np.pi / 2, atol=1e-10)

    def test_partial_overlap(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            subspace_angles,
        )

        # 한 축 공유, 한 축 직교
        A = np.eye(5)[:, [0, 1]]
        B = np.eye(5)[:, [0, 2]]
        angles = subspace_angles(A, B)
        # 첫 각도 0 (공유), 두 번째 π/2
        np.testing.assert_allclose(angles[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(angles[-1], np.pi / 2, atol=1e-10)

    def test_row_mismatch_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            subspace_angles,
        )

        with pytest.raises(ValueError, match="row count"):
            subspace_angles(np.zeros((5, 3)), np.zeros((4, 3)))

    def test_invalid_ndim(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            subspace_angles,
        )

        with pytest.raises(ValueError, match="2D"):
            subspace_angles(np.zeros(5), np.zeros((5, 3)))


class TestGrassmannDistance:
    def test_zero_for_identical(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            grassmann_distance,
        )

        A = np.eye(5)[:, :3]
        d = grassmann_distance(A, A)
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_max_for_orthogonal(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            grassmann_distance,
        )

        A = np.eye(5)[:, :2]
        B = np.eye(5)[:, 2:4]
        d = grassmann_distance(A, B)
        # √(2 · (π/2)²) = π/√2
        np.testing.assert_allclose(d, np.pi / np.sqrt(2), atol=1e-10)


class TestChordalDistance:
    def test_zero_identical(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            subspace_distance_chordal,
        )

        A = np.eye(5)[:, :3]
        d = subspace_distance_chordal(A, A)
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_orthogonal_max(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            subspace_distance_chordal,
        )

        A = np.eye(5)[:, :2]
        B = np.eye(5)[:, 2:4]
        d = subspace_distance_chordal(A, B)
        # sin(π/2) = 1, √2
        np.testing.assert_allclose(d, np.sqrt(2), atol=1e-10)


class TestAlignmentMatrix:
    def test_identical_diagonal(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            mode_alignment_matrix,
        )

        A = np.eye(5)[:, :3]
        sim = mode_alignment_matrix(A, A)
        np.testing.assert_allclose(np.diag(sim), 1.0)

    def test_orthogonal_zero(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            mode_alignment_matrix,
        )

        A = np.eye(5)[:, :2]
        B = np.eye(5)[:, 2:4]
        sim = mode_alignment_matrix(A, B)
        np.testing.assert_allclose(sim, 0.0, atol=1e-10)

    def test_row_mismatch_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            mode_alignment_matrix,
        )

        with pytest.raises(ValueError, match="row count"):
            mode_alignment_matrix(np.zeros((5, 3)), np.zeros((4, 3)))


class TestBestMatchAssignment:
    def test_diagonal(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            best_match_assignment,
        )

        # 정렬된 대각: A_i ↔ B_i
        sim = np.diag([0.9, 0.8, 0.7])
        out = best_match_assignment(sim)
        np.testing.assert_array_equal(out, [0, 1, 2])

    def test_swapped_modes(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            best_match_assignment,
        )

        # 모드 0과 1이 바뀌어 있음
        sim = np.array([[0.1, 0.9], [0.95, 0.2]])
        out = best_match_assignment(sim)
        assert out[1] == 0  # A_1 ↔ B_0
        assert out[0] == 1  # A_0 ↔ B_1

    def test_invalid_ndim_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            best_match_assignment,
        )

        with pytest.raises(ValueError, match="2D"):
            best_match_assignment(np.zeros(10))


class TestDriftScore:
    def test_no_drift(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            drift_score,
        )

        A = np.eye(5)[:, :3]
        d = drift_score(A, A)
        assert d < 1e-10

    def test_full_drift(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            drift_score,
        )

        A = np.eye(5)[:, :2]
        B = np.eye(5)[:, 2:4]
        d = drift_score(A, B)
        # max angle = π/2 → score = 1
        np.testing.assert_allclose(d, 1.0, atol=1e-10)


class TestProperOrthogonalBasis:
    def test_basic_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            proper_orthogonal_basis,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 30))
        basis, s = proper_orthogonal_basis(X, n_modes=5)
        # 자동 reshape: 더 큰 차원이 공간 (50)
        assert basis.shape == (50, 5)
        assert s.shape == (5,)
        # 직교
        np.testing.assert_allclose(basis.T @ basis, np.eye(5), atol=1e-10)

    def test_n_t_first(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            proper_orthogonal_basis,
        )

        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 50))  # n_t=30 < n_x=50
        basis, _ = proper_orthogonal_basis(X, n_modes=5)
        assert basis.shape == (50, 5)

    def test_invalid_ndim(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            proper_orthogonal_basis,
        )

        with pytest.raises(ValueError, match="2D"):
            proper_orthogonal_basis(np.zeros(50), n_modes=3)

    def test_no_centering(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_tracking import (
            proper_orthogonal_basis,
        )

        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 20)) + 5.0
        basis, _ = proper_orthogonal_basis(X, n_modes=3, center=False)
        assert basis.shape[0] == 40
