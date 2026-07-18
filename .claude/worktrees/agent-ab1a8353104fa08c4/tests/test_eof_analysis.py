"""Round 621 — EOF decomposition + reconstruction + North test + varimax rotation."""

from __future__ import annotations

import numpy as np
import pytest


def _make_data(n_t=200, n_x=40, n_true_modes=3, seed=0):
    rng = np.random.default_rng(seed)
    # 진짜 모드 패턴
    x = np.linspace(0, 1, n_x)
    patterns = np.column_stack([
        np.sin(np.pi * x),
        np.sin(2 * np.pi * x),
        np.sin(3 * np.pi * x),
    ])  # (n_x, 3)
    # 시간 계수 (서로 독립)
    pcs = rng.standard_normal((n_t, n_true_modes)) * np.array([3.0, 2.0, 1.0])
    X = pcs @ patterns.T + 0.1 * rng.standard_normal((n_t, n_x))
    return X


class TestEOFDecomposition:
    def test_basic_shapes(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition

        X = _make_data()
        eofs, pcs, var = eof_decomposition(X, n_modes=5)
        assert eofs.shape == (40, 5)
        assert pcs.shape == (200, 5)
        assert var.shape == (5,)

    def test_variance_decreasing(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition

        X = _make_data()
        _, _, var = eof_decomposition(X, n_modes=5)
        assert np.all(np.diff(var) <= 0)
        assert var[0] > 0

    def test_dominant_mode_captures_main_variance(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition

        X = _make_data(n_true_modes=3)
        _, _, var = eof_decomposition(X, n_modes=10)
        # 처음 3개가 변동의 대부분
        assert var[:3].sum() > 0.9

    def test_orthogonality(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition

        X = _make_data()
        eofs, _, _ = eof_decomposition(X, n_modes=5)
        # EOFs는 직교
        gram = eofs.T @ eofs
        np.testing.assert_allclose(gram, np.eye(5), atol=1e-8)

    def test_with_weights(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition

        X = _make_data()
        n_x = X.shape[1]
        w = np.ones(n_x)
        eofs, _, _ = eof_decomposition(X, n_modes=3, weights=w)
        assert eofs.shape == (n_x, 3)

    def test_standardize(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition

        rng = np.random.default_rng(1)
        # 일부 위치에서 분산이 매우 다름
        X = rng.standard_normal((100, 20))
        X[:, 0] *= 100.0  # 첫 위치 분산 매우 큼
        # 표준화 안 하면 첫 위치가 EOF1을 지배
        eofs1, _, var1 = eof_decomposition(X, n_modes=3, standardize=False)
        eofs2, _, var2 = eof_decomposition(X, n_modes=3, standardize=True)
        # 표준화하면 EOF1이 더 분산됨
        assert abs(eofs1[0, 0]) > 0.5

    def test_sign_convention(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition

        X = _make_data()
        eofs, _, _ = eof_decomposition(X, n_modes=3, sign_convention=True)
        # 최대 절댓값 위치에서 양수
        for k in range(3):
            idx_max = int(np.argmax(np.abs(eofs[:, k])))
            assert eofs[idx_max, k] >= 0

    def test_invalid_X_shape(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition

        with pytest.raises(ValueError, match="2D"):
            eof_decomposition(np.zeros(50), n_modes=3)

    def test_invalid_weights_shape(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition

        X = _make_data()
        with pytest.raises(ValueError, match="weights shape"):
            eof_decomposition(X, n_modes=3, weights=np.ones(99))

    def test_negative_weights_raises(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition

        X = _make_data()
        w = np.ones(X.shape[1])
        w[0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            eof_decomposition(X, n_modes=3, weights=w)

    def test_invalid_n_modes(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition

        X = _make_data()
        with pytest.raises(ValueError, match="n_modes"):
            eof_decomposition(X, n_modes=0)


class TestReconstruct:
    def test_full_reconstruction(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import (
            eof_decomposition,
            reconstruct_from_eof,
        )

        X = _make_data()
        n_t = X.shape[0]
        eofs, pcs, _ = eof_decomposition(X, n_modes=min(n_t, X.shape[1]))
        rec = reconstruct_from_eof(eofs, pcs, mean=X.mean(axis=0))
        # 모든 모드 사용 → 완전 재구성
        np.testing.assert_allclose(rec, X, atol=1e-8)

    def test_truncated_loss(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import (
            eof_decomposition,
            reconstruct_from_eof,
        )

        X = _make_data()
        eofs, pcs, _ = eof_decomposition(X, n_modes=2)
        rec = reconstruct_from_eof(eofs, pcs, mean=X.mean(axis=0))
        err = np.linalg.norm(X - rec)
        assert err > 0  # 절단으로 손실

    def test_no_mean(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import reconstruct_from_eof

        eofs = np.array([[1.0, 0.0], [0.0, 1.0]])
        pcs = np.array([[1.0, 0.0], [0.0, 1.0]])
        rec = reconstruct_from_eof(eofs, pcs)
        np.testing.assert_allclose(rec, np.eye(2))

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import reconstruct_from_eof

        with pytest.raises(ValueError, match="n_modes"):
            reconstruct_from_eof(np.zeros((10, 3)), np.zeros((20, 5)))


class TestNorthTest:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import (
            north_significance_test,
        )

        var = np.array([0.5, 0.3, 0.1, 0.05])
        se = north_significance_test(var, n_t=200)
        assert se.shape == (4,)
        assert np.all(se > 0)

    def test_invalid_n_t(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import (
            north_significance_test,
        )

        with pytest.raises(ValueError, match="n_t"):
            north_significance_test(np.array([0.5]), n_t=1)


class TestVarimax:
    def test_returns_same_shape(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import (
            eof_decomposition,
            varimax_rotation,
        )

        X = _make_data()
        eofs, _, _ = eof_decomposition(X, n_modes=3)
        rotated = varimax_rotation(eofs, n_iter=50)
        assert rotated.shape == eofs.shape

    def test_single_mode_returns_unchanged(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import varimax_rotation

        eofs = np.random.default_rng(0).standard_normal((10, 1))
        rotated = varimax_rotation(eofs)
        np.testing.assert_allclose(rotated, eofs)

    def test_invalid_shape_raises(self) -> None:
        from naviertwin.core.flow_analysis.eof_analysis import varimax_rotation

        with pytest.raises(ValueError, match="2D"):
            varimax_rotation(np.zeros(10))
