"""R649 — time-series similarity: SBD, MASS, motifs, template matching."""

from __future__ import annotations

import numpy as np
import pytest


class TestSBD:
    def test_zero_for_identical(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import shape_based_distance

        x = np.sin(np.linspace(0, 2 * np.pi, 100))
        d = shape_based_distance(x, x)
        assert d < 1e-6

    def test_zero_for_scaled(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import shape_based_distance

        x = np.sin(np.linspace(0, 2 * np.pi, 100))
        # z-norm 후 동일
        d = shape_based_distance(x, 5 * x + 100)
        assert d < 1e-6

    def test_high_for_different(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import shape_based_distance

        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        y = rng.standard_normal(200)
        d = shape_based_distance(x, y)
        assert d > 0.3

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import shape_based_distance

        with pytest.raises(ValueError, match="shape"):
            shape_based_distance(np.zeros(10), np.zeros(20))

    def test_too_short_raises(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import shape_based_distance

        with pytest.raises(ValueError, match="2 points"):
            shape_based_distance(np.array([1.0]), np.array([1.0]))

    def test_constant_input_returns_distance(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import shape_based_distance

        d = shape_based_distance(np.zeros(50), np.ones(50))
        assert d > 0


class TestMASS:
    def test_finds_known_pattern(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import mass_search

        rng = np.random.default_rng(1)
        series = rng.standard_normal(500)
        # 100 위치에 query를 심음
        query = series[100:130].copy()
        dist = mass_search(query, series)
        # 가장 작은 거리가 100 위치
        assert int(np.argmin(dist)) == 100
        assert dist[100] < 1e-6

    def test_query_too_long_raises(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import mass_search

        with pytest.raises(ValueError, match="series length"):
            mass_search(np.zeros(50), np.zeros(20))

    def test_query_too_short_raises(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import mass_search

        with pytest.raises(ValueError, match="query length"):
            mass_search(np.array([1.0]), np.zeros(100))

    def test_distance_profile_length(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import mass_search

        rng = np.random.default_rng(2)
        series = rng.standard_normal(200)
        query = rng.standard_normal(20)
        dist = mass_search(query, series)
        assert dist.shape == (200 - 20 + 1,)
        assert np.all(dist >= 0)


class TestFindMotifs:
    def test_returns_pairs(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import find_top_k_motifs

        rng = np.random.default_rng(3)
        # 두 위치에 동일 패턴 심기
        series = rng.standard_normal(300)
        pattern = np.sin(np.linspace(0, 2 * np.pi, 30))
        series[50:80] = pattern
        series[200:230] = pattern
        motifs = find_top_k_motifs(series, window=30, k=1)
        assert len(motifs) >= 1
        a, b, d = motifs[0]
        # 두 위치 매칭
        assert {a, b} == {50, 200} or abs(a - 50) < 5 or abs(b - 200) < 5

    def test_invalid_window(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import find_top_k_motifs

        with pytest.raises(ValueError, match="window"):
            find_top_k_motifs(np.zeros(100), window=1, k=1)

    def test_invalid_k(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import find_top_k_motifs

        with pytest.raises(ValueError, match="k"):
            find_top_k_motifs(np.zeros(100), window=10, k=0)


class TestTemplateMatching:
    def test_finds_repeated_pattern(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import template_matching

        # 0과 50 위치에 동일 template
        template = np.sin(np.linspace(0, 2 * np.pi, 20))
        series = np.zeros(200)
        series[0:20] = template
        series[50:70] = template
        matches = template_matching(template, series, threshold=0.1)
        assert 0 in matches
        assert 50 in matches

    def test_no_match_above_threshold(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import template_matching

        rng = np.random.default_rng(4)
        template = rng.standard_normal(20)
        series = rng.standard_normal(200)
        # 매우 작은 임계값 → 매칭 없음
        matches = template_matching(template, series, threshold=0.001)
        assert len(matches) == 0

    def test_invalid_threshold(self) -> None:
        from naviertwin.core.flow_analysis.ts_similarity import template_matching

        with pytest.raises(ValueError, match="threshold"):
            template_matching(np.zeros(20), np.zeros(100), threshold=0.0)
