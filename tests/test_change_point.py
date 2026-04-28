"""Round 638 — change-point detection: Binary Segmentation, PELT, Window."""

from __future__ import annotations

import numpy as np
import pytest


class TestBinarySegmentation:
    def test_one_change(self) -> None:
        from naviertwin.core.flow_analysis.change_point import binary_segmentation

        rng = np.random.default_rng(0)
        x = np.concatenate([rng.standard_normal(60), 5 + rng.standard_normal(60)])
        cps = binary_segmentation(x, n_changepoints=1)
        assert len(cps) == 1
        assert abs(cps[0] - 60) < 5

    def test_two_changes(self) -> None:
        from naviertwin.core.flow_analysis.change_point import binary_segmentation

        rng = np.random.default_rng(1)
        x = np.concatenate([
            rng.standard_normal(40),
            5 + rng.standard_normal(40),
            -3 + rng.standard_normal(40),
        ])
        cps = binary_segmentation(x, n_changepoints=2)
        assert len(cps) == 2
        # 가까운 변화점 (40, 80)
        assert abs(cps[0] - 40) < 8
        assert abs(cps[1] - 80) < 8

    def test_too_short_raises(self) -> None:
        from naviertwin.core.flow_analysis.change_point import binary_segmentation

        with pytest.raises(ValueError, match="too short"):
            binary_segmentation(np.zeros(5), n_changepoints=1, min_size=5)

    def test_invalid_n_cps(self) -> None:
        from naviertwin.core.flow_analysis.change_point import binary_segmentation

        with pytest.raises(ValueError, match="n_changepoints"):
            binary_segmentation(np.zeros(50), n_changepoints=0)

    def test_invalid_min_size(self) -> None:
        from naviertwin.core.flow_analysis.change_point import binary_segmentation

        with pytest.raises(ValueError, match="min_size"):
            binary_segmentation(np.zeros(50), n_changepoints=1, min_size=0)

    def test_constant_signal_no_change(self) -> None:
        from naviertwin.core.flow_analysis.change_point import binary_segmentation

        x = np.ones(50)
        cps = binary_segmentation(x, n_changepoints=2)
        # gain ≤ 0 → 검출 안 됨
        assert len(cps) == 0


class TestPELT:
    def test_one_change(self) -> None:
        from naviertwin.core.flow_analysis.change_point import pelt

        rng = np.random.default_rng(2)
        x = np.concatenate([rng.standard_normal(80), 8 + rng.standard_normal(80)])
        cps = pelt(x, penalty=10.0)
        assert len(cps) >= 1
        # 80 근처 변화점
        nearest = min(cps, key=lambda c: abs(c - 80))
        assert abs(nearest - 80) < 10

    def test_no_changes_for_constant_with_high_penalty(self) -> None:
        from naviertwin.core.flow_analysis.change_point import pelt

        rng = np.random.default_rng(3)
        x = rng.standard_normal(100)
        cps = pelt(x, penalty=1000.0)
        # 매우 높은 패널티 → 변화점 없음
        assert len(cps) == 0

    def test_default_penalty(self) -> None:
        from naviertwin.core.flow_analysis.change_point import pelt

        rng = np.random.default_rng(4)
        x = rng.standard_normal(50)
        # penalty=None → BIC 기반
        cps = pelt(x)
        assert isinstance(cps, list)

    def test_too_short_raises(self) -> None:
        from naviertwin.core.flow_analysis.change_point import pelt

        with pytest.raises(ValueError, match="too short"):
            pelt(np.zeros(5), min_size=5)

    def test_invalid_min_size(self) -> None:
        from naviertwin.core.flow_analysis.change_point import pelt

        with pytest.raises(ValueError, match="min_size"):
            pelt(np.zeros(50), min_size=0)


class TestWindowMethod:
    def test_clear_change_detected(self) -> None:
        from naviertwin.core.flow_analysis.change_point import window_method

        rng = np.random.default_rng(5)
        x = np.concatenate([rng.standard_normal(50), 10 + rng.standard_normal(50)])
        cps = window_method(x, width=10, threshold=2.0)
        assert len(cps) >= 1
        # 50 근처
        nearest = min(cps, key=lambda c: abs(c - 50))
        assert abs(nearest - 50) < 10

    def test_no_change_high_threshold(self) -> None:
        from naviertwin.core.flow_analysis.change_point import window_method

        rng = np.random.default_rng(6)
        x = rng.standard_normal(100)
        cps = window_method(x, width=10, threshold=5.0)
        assert len(cps) == 0

    def test_invalid_width(self) -> None:
        from naviertwin.core.flow_analysis.change_point import window_method

        with pytest.raises(ValueError, match="width"):
            window_method(np.zeros(50), width=0)

    def test_invalid_threshold(self) -> None:
        from naviertwin.core.flow_analysis.change_point import window_method

        with pytest.raises(ValueError, match="threshold"):
            window_method(np.zeros(50), width=5, threshold=0.0)


class TestSegmentMeans:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.change_point import segment_means

        x = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 9.0, 9.0])
        means = segment_means(x, [3, 6])
        np.testing.assert_allclose(means, [1.0, 5.0, 9.0])

    def test_empty_input(self) -> None:
        from naviertwin.core.flow_analysis.change_point import segment_means

        assert segment_means(np.array([]), []) == []

    def test_no_changepoints(self) -> None:
        from naviertwin.core.flow_analysis.change_point import segment_means

        x = np.array([1.0, 2.0, 3.0])
        means = segment_means(x, [])
        np.testing.assert_allclose(means, [2.0])


class TestDetectionScore:
    def test_perfect_segmentation(self) -> None:
        from naviertwin.core.flow_analysis.change_point import detection_score

        x = np.array([1.0] * 10 + [5.0] * 10)
        # 분명한 변화점 10에서 정확히 분할
        score = detection_score(x, [10])
        assert score > 0.99

    def test_no_changepoints_zero(self) -> None:
        from naviertwin.core.flow_analysis.change_point import detection_score

        rng = np.random.default_rng(7)
        x = rng.standard_normal(100)
        score = detection_score(x, [])
        assert score == 0.0

    def test_constant_signal(self) -> None:
        from naviertwin.core.flow_analysis.change_point import detection_score

        x = np.ones(50)
        score = detection_score(x, [25])
        assert score == 0.0

    def test_short_returns_zero(self) -> None:
        from naviertwin.core.flow_analysis.change_point import detection_score

        score = detection_score(np.array([1.0]), [])
        assert score == 0.0
