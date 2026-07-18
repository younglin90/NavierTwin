"""Round 631 — ROM a posteriori error + certification metrics."""

from __future__ import annotations

import numpy as np
import pytest


class TestReconstructionResidual:
    def test_full_basis_zero(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            reconstruction_residual,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 10))
        # 완전 기저 → 잔차 0
        U, s, _ = np.linalg.svd(X.T, full_matrices=False)
        residuals = reconstruction_residual(X, U)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)

    def test_truncated_nonzero(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            reconstruction_residual,
        )

        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 20))
        U, s, _ = np.linalg.svd(X.T, full_matrices=False)
        # rank 5만 사용 → 잔차 > 0
        residuals = reconstruction_residual(X, U[:, :5])
        assert np.all(residuals > 0)

    def test_1d_input(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            reconstruction_residual,
        )

        rng = np.random.default_rng(2)
        x = rng.standard_normal(15)
        # 단일 모드: x를 정규화한 것
        v = x / np.linalg.norm(x)
        V = v[:, None]
        residual = reconstruction_residual(x, V)
        assert isinstance(residual, float)
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)

    def test_invalid_shape_1d(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            reconstruction_residual,
        )

        with pytest.raises(ValueError, match="X length"):
            reconstruction_residual(np.zeros(10), np.zeros((20, 5)))

    def test_invalid_shape_2d(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            reconstruction_residual,
        )

        with pytest.raises(ValueError, match="X cols"):
            reconstruction_residual(np.zeros((10, 8)), np.zeros((20, 5)))

    def test_basis_not_2d(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            reconstruction_residual,
        )

        with pytest.raises(ValueError, match="2D"):
            reconstruction_residual(np.zeros((10, 5)), np.zeros(5))


class TestRelativeResidual:
    def test_value_in_range(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            relative_residual,
        )

        rng = np.random.default_rng(3)
        X = rng.standard_normal((20, 10))
        U, _, _ = np.linalg.svd(X.T, full_matrices=False)
        rel = relative_residual(X, U[:, :3])
        assert np.all((0 <= rel) & (rel <= 1))

    def test_full_basis_zero(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            relative_residual,
        )

        rng = np.random.default_rng(4)
        X = rng.standard_normal((10, 5))
        U, _, _ = np.linalg.svd(X.T, full_matrices=False)
        rel = relative_residual(X, U)
        np.testing.assert_allclose(rel, 0.0, atol=1e-10)


class TestLOOScore:
    def test_basic(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            leave_one_out_score,
        )

        rng = np.random.default_rng(5)
        X = rng.standard_normal((20, 15))
        score = leave_one_out_score(X, n_modes=5)
        assert "loo_mse" in score
        assert "loo_max" in score
        assert "loo_mean_rel" in score
        assert score["loo_mse"] >= 0

    def test_more_modes_smaller_error(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            leave_one_out_score,
        )

        rng = np.random.default_rng(6)
        # 진짜 rank 3 데이터
        U_true = rng.standard_normal((20, 3))
        V_true = rng.standard_normal((3, 15))
        X = U_true @ V_true + 0.01 * rng.standard_normal((20, 15))
        s_low = leave_one_out_score(X, n_modes=2)
        s_high = leave_one_out_score(X, n_modes=8)
        # 더 많은 모드 → 더 작은 LOO 오차 (진짜 rank 3 이하면 동일)
        assert s_high["loo_mean_rel"] <= s_low["loo_mean_rel"] + 0.05

    def test_too_few_samples(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            leave_one_out_score,
        )

        with pytest.raises(ValueError, match="n_t"):
            leave_one_out_score(np.zeros((2, 10)), n_modes=2)

    def test_invalid_n_modes(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            leave_one_out_score,
        )

        with pytest.raises(ValueError, match="n_modes"):
            leave_one_out_score(np.zeros((10, 5)), n_modes=0)

    def test_1d_X_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            leave_one_out_score,
        )

        with pytest.raises(ValueError, match="2D"):
            leave_one_out_score(np.zeros(10), n_modes=2)


class TestCoefficientEnvelope:
    def test_in_range_zero_violation(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            coefficient_envelope,
        )

        rng = np.random.default_rng(7)
        C = rng.standard_normal((100, 5))
        new = C.mean(axis=0)  # 학습 평균
        env = coefficient_envelope(C, new)
        assert env["max_z"] < 0.1
        assert env["bbox_violation_count"] == 0

    def test_far_outside_high_z(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            coefficient_envelope,
        )

        rng = np.random.default_rng(8)
        C = rng.standard_normal((100, 5))
        # 학습 범위 훨씬 벗어남
        new = C.mean(axis=0) + 10.0 * C.std(axis=0)
        env = coefficient_envelope(C, new)
        assert env["max_z"] > 5.0
        assert env["bbox_violation_count"] == 5

    def test_invalid_C_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            coefficient_envelope,
        )

        with pytest.raises(ValueError, match="2D"):
            coefficient_envelope(np.zeros(10), np.zeros(5))

    def test_invalid_coeff_length(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            coefficient_envelope,
        )

        with pytest.raises(ValueError, match="length"):
            coefficient_envelope(np.zeros((10, 5)), np.zeros(3))


class TestProjectionErrorBound:
    def test_full_rank_zero(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            projection_error_bound,
        )

        s = np.array([3.0, 2.0, 1.0])
        bound = projection_error_bound(s, n_modes=3)
        assert bound == 0.0

    def test_zero_rank_full_norm(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            projection_error_bound,
        )

        s = np.array([3.0, 2.0, 1.0])
        bound = projection_error_bound(s, n_modes=0)
        # √(9 + 4 + 1) = √14
        np.testing.assert_allclose(bound, np.sqrt(14.0))

    def test_invalid_n_modes(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            projection_error_bound,
        )

        with pytest.raises(ValueError, match="n_modes"):
            projection_error_bound(np.ones(5), n_modes=10)


class TestConfidenceScore:
    def test_below_threshold_one(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            confidence_score,
        )

        s = confidence_score(residual_norm=0.01, reference_norm=1.0, threshold=0.05)
        assert s == 1.0

    def test_above_threshold_decay(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            confidence_score,
        )

        s = confidence_score(residual_norm=0.5, reference_norm=1.0, threshold=0.05)
        assert 0 < s < 1.0

    def test_invalid_threshold(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            confidence_score,
        )

        with pytest.raises(ValueError, match="threshold"):
            confidence_score(0.1, 1.0, threshold=0.0)

    def test_invalid_reference(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_certification import (
            confidence_score,
        )

        with pytest.raises(ValueError, match="reference_norm"):
            confidence_score(0.1, 0.0, threshold=0.05)
