"""Round 642 — Gappy POD reconstruction (Everson-Sirovich 1995)."""

from __future__ import annotations

import numpy as np
import pytest


class TestGappyReconstruct:
    def test_full_mask_recovers_exact(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            gappy_reconstruct,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 30))
        U, _, _ = np.linalg.svd(X, full_matrices=False)
        V = U[:, :30]  # 완전 기저
        x = X[:, 0]
        mask = np.ones(50, dtype=bool)
        rec = gappy_reconstruct(V, x, mask)
        np.testing.assert_allclose(rec, x, atol=1e-10)

    def test_partial_mask(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            gappy_reconstruct,
        )

        rng = np.random.default_rng(1)
        # rank 5 데이터
        U_true = rng.standard_normal((50, 5))
        V_true = rng.standard_normal((5, 20))
        X = U_true @ V_true
        # 진짜 기저
        U, _, _ = np.linalg.svd(X, full_matrices=False)
        V = U[:, :5]
        mask = np.ones(50, dtype=bool)
        mask[10:30] = False  # 60% 관측
        rec = gappy_reconstruct(V, X[:, 0], mask)
        # rank가 정확하면 완전 복원
        np.testing.assert_allclose(rec, X[:, 0], atol=1e-8)

    def test_2d_input(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            gappy_reconstruct,
        )

        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 5))
        U, _, _ = np.linalg.svd(X, full_matrices=False)
        V = U
        mask = np.ones(30, dtype=bool)
        rec = gappy_reconstruct(V, X, mask)
        np.testing.assert_allclose(rec, X, atol=1e-10)

    def test_no_observations_returns_zero(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            gappy_reconstruct,
        )

        V = np.random.default_rng(3).standard_normal((20, 4))
        x = np.random.default_rng(4).standard_normal(20)
        mask = np.zeros(20, dtype=bool)
        rec = gappy_reconstruct(V, x, mask)
        np.testing.assert_array_equal(rec, np.zeros(20))

    def test_invalid_basis_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            gappy_reconstruct,
        )

        with pytest.raises(ValueError, match="2D"):
            gappy_reconstruct(np.zeros(20), np.zeros(20), np.ones(20, dtype=bool))

    def test_mask_shape_mismatch(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            gappy_reconstruct,
        )

        with pytest.raises(ValueError, match="mask"):
            gappy_reconstruct(
                np.zeros((20, 5)), np.zeros(20), np.ones(15, dtype=bool),
            )

    def test_partial_length_mismatch(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            gappy_reconstruct,
        )

        with pytest.raises(ValueError, match="partial"):
            gappy_reconstruct(
                np.zeros((20, 5)), np.zeros(15), np.ones(20, dtype=bool),
            )

    def test_partial_3d_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            gappy_reconstruct,
        )

        with pytest.raises(ValueError, match="1D or 2D"):
            gappy_reconstruct(
                np.zeros((20, 5)),
                np.zeros((20, 3, 2)),
                np.ones(20, dtype=bool),
            )


class TestGappyCoefficients:
    def test_basic(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            gappy_coefficients,
        )

        rng = np.random.default_rng(5)
        V = rng.standard_normal((30, 4))
        Q, _ = np.linalg.qr(V)
        # 진짜 계수
        alpha_true = np.array([1.0, 2.0, 3.0, 4.0])
        x = Q @ alpha_true
        mask = np.ones(30, dtype=bool)
        alpha = gappy_coefficients(Q, x, mask)
        np.testing.assert_allclose(alpha, alpha_true, atol=1e-10)

    def test_no_observations_zero(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            gappy_coefficients,
        )

        V = np.random.default_rng(6).standard_normal((20, 3))
        x = np.zeros(20)
        mask = np.zeros(20, dtype=bool)
        alpha = gappy_coefficients(V, x, mask)
        np.testing.assert_array_equal(alpha, np.zeros(3))

    def test_invalid_basis_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            gappy_coefficients,
        )

        with pytest.raises(ValueError, match="2D"):
            gappy_coefficients(np.zeros(10), np.zeros(10), np.ones(10, dtype=bool))

    def test_shape_mismatch(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            gappy_coefficients,
        )

        with pytest.raises(ValueError, match="partial/mask"):
            gappy_coefficients(
                np.zeros((10, 3)), np.zeros(8), np.ones(10, dtype=bool),
            )


class TestGappyIter:
    def test_recovers_low_rank(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import gappy_iter

        rng = np.random.default_rng(7)
        # 진짜 rank 3
        U_true = rng.standard_normal((40, 3))
        V_true = rng.standard_normal((3, 20))
        X_true = U_true @ V_true
        # 30% 결측
        mask = rng.uniform(0, 1, X_true.shape) > 0.3
        X = X_true.copy()
        X[~mask] = 0.0  # 결측 → 0
        X_filled = gappy_iter(X, mask, n_modes=5, max_iter=30)
        # 결측 영역의 오차가 작음
        rec_err = np.sqrt(np.mean((X_filled[~mask] - X_true[~mask]) ** 2))
        true_norm = np.sqrt(np.mean(X_true[~mask] ** 2))
        assert rec_err / max(true_norm, 1e-30) < 0.5

    def test_1d_mask_broadcasts(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import gappy_iter

        rng = np.random.default_rng(8)
        X = rng.standard_normal((30, 10))
        mask = np.ones(30, dtype=bool)
        mask[5:10] = False
        X_filled = gappy_iter(X, mask, n_modes=3, max_iter=10)
        assert X_filled.shape == X.shape

    def test_invalid_X_ndim(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import gappy_iter

        with pytest.raises(ValueError, match="2D"):
            gappy_iter(np.zeros(20), np.ones(20, dtype=bool), n_modes=3)

    def test_invalid_mask_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import gappy_iter

        with pytest.raises(ValueError, match="mask"):
            gappy_iter(
                np.zeros((10, 5)),
                np.ones((8, 5), dtype=bool),
                n_modes=3,
            )

    def test_invalid_n_modes(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import gappy_iter

        with pytest.raises(ValueError, match="n_modes"):
            gappy_iter(
                np.zeros((10, 5)),
                np.ones((10, 5), dtype=bool),
                n_modes=0,
            )


class TestReconstructionError:
    def test_zero_for_perfect(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            reconstruction_error,
        )

        X_true = np.random.default_rng(9).standard_normal((10, 5))
        X_filled = X_true.copy()
        mask = np.ones((10, 5), dtype=bool)
        mask[2:5, :] = False  # 결측
        err = reconstruction_error(X_true, X_filled, mask)
        assert err == 0.0

    def test_no_missing_zero(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            reconstruction_error,
        )

        X_true = np.random.default_rng(10).standard_normal((10, 5))
        X_filled = X_true + 1.0
        mask = np.ones((10, 5), dtype=bool)  # 모두 관측 → missing 없음
        err = reconstruction_error(X_true, X_filled, mask)
        assert err == 0.0

    def test_shape_mismatch(self) -> None:
        from naviertwin.core.dimensionality_reduction.gappy_pod import (
            reconstruction_error,
        )

        with pytest.raises(ValueError, match="shape"):
            reconstruction_error(
                np.zeros((10, 5)),
                np.zeros((10, 4)),
                np.ones((10, 5), dtype=bool),
            )
