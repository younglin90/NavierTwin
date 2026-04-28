"""Round 639 — multivariate anomaly scores: Mahalanobis, LOF, IF, z-score, Hampel."""

from __future__ import annotations

import numpy as np
import pytest


class TestMahalanobis:
    def test_outlier_high_score(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import mahalanobis_score

        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 3))
        X[0] = [10.0, 10.0, 10.0]
        scores = mahalanobis_score(X)
        # 첫 점이 가장 큰 점수
        assert int(np.argmax(scores)) == 0

    def test_with_reference(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import mahalanobis_score

        rng = np.random.default_rng(1)
        ref = rng.standard_normal((200, 2))
        # 새 점 (학습 분포 밖)
        new = np.array([[5.0, 5.0], [0.0, 0.0]])
        scores = mahalanobis_score(new, reference=ref)
        # 0번 점이 더 큼
        assert scores[0] > scores[1]

    def test_invalid_X_ndim(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import mahalanobis_score

        with pytest.raises(ValueError, match="2D"):
            mahalanobis_score(np.zeros(10))

    def test_reference_shape_mismatch(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import mahalanobis_score

        with pytest.raises(ValueError, match="reference"):
            mahalanobis_score(np.zeros((10, 3)), reference=np.zeros((10, 4)))


class TestLOF:
    def test_outlier_lof_above_one(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import lof_score

        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 2))
        # 명백한 outlier 추가
        X = np.vstack([X, [[10.0, 10.0]]])
        scores = lof_score(X, k=5)
        # 마지막 점 (outlier)이 가장 큼
        assert int(np.argmax(scores)) == X.shape[0] - 1
        assert scores[-1] > 1.0

    def test_invalid_X_ndim(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import lof_score

        with pytest.raises(ValueError, match="2D"):
            lof_score(np.zeros(50))

    def test_invalid_k(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import lof_score

        with pytest.raises(ValueError, match="k"):
            lof_score(np.zeros((10, 3)), k=10)


class TestIsolationDepth:
    def test_outlier_low_depth(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import isolation_depth

        rng = np.random.default_rng(3)
        X = rng.standard_normal((100, 2))
        X = np.vstack([X, [[20.0, 20.0]]])
        depths = isolation_depth(X, n_trees=20, sample_size=50, seed=0)
        # outlier가 더 얕게 분리됨
        assert depths[-1] < depths[:100].mean()

    def test_invalid_X_ndim(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import isolation_depth

        with pytest.raises(ValueError, match="2D"):
            isolation_depth(np.zeros(50))

    def test_invalid_n_trees(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import isolation_depth

        with pytest.raises(ValueError, match="n_trees"):
            isolation_depth(np.zeros((10, 2)), n_trees=0)


class TestZScoreMax:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import z_score_max

        rng = np.random.default_rng(4)
        X = rng.standard_normal((100, 3))
        X[0] = [10.0, 10.0, 10.0]
        scores = z_score_max(X)
        assert int(np.argmax(scores)) == 0

    def test_invalid_ndim(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import z_score_max

        with pytest.raises(ValueError, match="2D"):
            z_score_max(np.zeros(10))


class TestHampelScore1D:
    def test_spike_high_score(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import hampel_score_1d

        rng = np.random.default_rng(0)
        x = 1.0 + 0.05 * rng.standard_normal(50)
        x[25] = 100.0
        scores = hampel_score_1d(x, window=5)
        assert scores[25] > 10.0
        # 다른 점들은 낮음
        assert scores[10] < 5.0

    def test_invalid_window_even(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import hampel_score_1d

        with pytest.raises(ValueError, match="window"):
            hampel_score_1d(np.zeros(50), window=4)

    def test_invalid_window_too_small(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import hampel_score_1d

        with pytest.raises(ValueError, match="window"):
            hampel_score_1d(np.zeros(50), window=1)

    def test_constant_signal_zero(self) -> None:
        from naviertwin.core.flow_analysis.anomaly_score import hampel_score_1d

        x = np.ones(20)
        scores = hampel_score_1d(x, window=5)
        np.testing.assert_array_equal(scores, np.zeros(20))
