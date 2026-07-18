"""Round 637 — POD coefficient trajectory clustering."""

from __future__ import annotations

import numpy as np
import pytest


class TestWindowKmeans:
    def test_basic_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            window_kmeans,
        )

        rng = np.random.default_rng(0)
        coeffs = rng.standard_normal((100, 5))
        labels, centers = window_kmeans(coeffs, window=20, n_clusters=3)
        assert labels.shape == (81,)  # 100 - 20 + 1
        assert centers.shape == (3, 5)

    def test_two_regimes_separated(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            window_kmeans,
        )

        rng = np.random.default_rng(1)
        # 처음 50: 평균 [0, 0]; 나머지 50: 평균 [10, 10]
        c1 = rng.standard_normal((50, 2)) + 0.0
        c2 = rng.standard_normal((50, 2)) + 10.0
        coeffs = np.vstack([c1, c2])
        labels, centers = window_kmeans(coeffs, window=10, n_clusters=2)
        # 두 모드가 잘 분리됨
        first_window_label = labels[5]
        last_window_label = labels[-5]
        assert first_window_label != last_window_label

    def test_invalid_coeffs_ndim(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            window_kmeans,
        )

        with pytest.raises(ValueError, match="2D"):
            window_kmeans(np.zeros(50), window=5, n_clusters=2)

    def test_invalid_window(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            window_kmeans,
        )

        with pytest.raises(ValueError, match="window"):
            window_kmeans(np.zeros((50, 3)), window=0, n_clusters=2)

    def test_invalid_n_clusters(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            window_kmeans,
        )

        with pytest.raises(ValueError, match="n_clusters"):
            window_kmeans(np.zeros((50, 3)), window=10, n_clusters=0)


class TestTrajectoryDistance:
    def test_euclidean_avg_zero_for_same(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            trajectory_distance_matrix,
        )

        t1 = np.zeros((10, 3))
        t2 = np.zeros((10, 3))
        D = trajectory_distance_matrix([t1, t2], metric="euclidean_avg")
        assert D.shape == (2, 2)
        np.testing.assert_allclose(D, np.zeros((2, 2)))

    def test_endpoint_metric(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            trajectory_distance_matrix,
        )

        t1 = np.zeros((10, 2))
        t2 = np.zeros((10, 2))
        t2[-1] = [3.0, 4.0]
        D = trajectory_distance_matrix([t1, t2], metric="endpoint")
        np.testing.assert_allclose(D[0, 1], 5.0)

    def test_frobenius_metric(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            trajectory_distance_matrix,
        )

        t1 = np.zeros((5, 3))
        t2 = np.ones((5, 3))
        D = trajectory_distance_matrix([t1, t2], metric="frobenius")
        np.testing.assert_allclose(D[0, 1], np.sqrt(15))

    def test_frobenius_shape_mismatch(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            trajectory_distance_matrix,
        )

        with pytest.raises(ValueError, match="frobenius"):
            trajectory_distance_matrix(
                [np.zeros((5, 3)), np.zeros((4, 3))], metric="frobenius",
            )

    def test_invalid_metric(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            trajectory_distance_matrix,
        )

        with pytest.raises(ValueError, match="metric"):
            trajectory_distance_matrix([np.zeros((5, 3))], metric="bogus")

    def test_empty(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            trajectory_distance_matrix,
        )

        D = trajectory_distance_matrix([])
        assert D.shape == (0, 0)


class TestSilhouette:
    def test_perfect_clusters_close_to_one(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            cluster_silhouette,
        )

        # 완전 분리된 두 클러스터
        X = np.vstack([
            np.zeros((20, 2)),
            np.full((20, 2), 100.0),
        ])
        labels = np.array([0] * 20 + [1] * 20)
        s = cluster_silhouette(X, labels)
        assert s > 0.9

    def test_random_clusters_low(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            cluster_silhouette,
        )

        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 3))
        labels = rng.integers(0, 3, size=50)
        s = cluster_silhouette(X, labels)
        # 무작위 라벨 → 낮음
        assert s < 0.3

    def test_single_cluster_zero(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            cluster_silhouette,
        )

        X = np.random.default_rng(3).standard_normal((10, 2))
        labels = np.zeros(10, dtype=np.intp)
        s = cluster_silhouette(X, labels)
        assert s == 0.0

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            cluster_silhouette,
        )

        with pytest.raises(ValueError, match="length mismatch"):
            cluster_silhouette(np.zeros((10, 2)), np.zeros(5, dtype=int))


class TestLabelRuns:
    def test_basic(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            label_runs,
        )

        labels = np.array([0, 0, 1, 1, 1, 0, 2, 2])
        runs = label_runs(labels)
        # (label, start, end)
        assert runs == [(0, 0, 2), (1, 2, 5), (0, 5, 6), (2, 6, 8)]

    def test_single_run(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            label_runs,
        )

        labels = np.zeros(10, dtype=np.intp)
        runs = label_runs(labels)
        assert runs == [(0, 0, 10)]

    def test_empty(self) -> None:
        from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
            label_runs,
        )

        runs = label_runs(np.array([], dtype=np.intp))
        assert runs == []
