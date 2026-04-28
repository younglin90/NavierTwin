"""Round 623 — conditional sampling: triggers, quadrants, event durations."""

from __future__ import annotations

import numpy as np
import pytest


class TestThresholdCrossings:
    def test_rising(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            find_threshold_crossings,
        )

        s = np.array([0.0, 1.0, 0.5, 2.0, 1.5, 3.0])
        idx = find_threshold_crossings(s, threshold=1.0, direction="rising")
        # 인덱스 1, 3, 5에서 1.0 이상으로 진입
        # transitions: above[i+1] - above[i]
        # above = [F, T, F, T, T, T] → diff = [1, -1, 1, 0, 0]
        np.testing.assert_array_equal(idx, [1, 3])

    def test_falling(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            find_threshold_crossings,
        )

        s = np.array([2.0, 0.5, 2.0, 0.5])
        idx = find_threshold_crossings(s, threshold=1.0, direction="falling")
        np.testing.assert_array_equal(idx, [1, 3])

    def test_both(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            find_threshold_crossings,
        )

        s = np.array([0.0, 2.0, 0.0])
        idx = find_threshold_crossings(s, threshold=1.0, direction="both")
        np.testing.assert_array_equal(idx, [1, 2])

    def test_invalid_direction(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            find_threshold_crossings,
        )

        with pytest.raises(ValueError, match="direction"):
            find_threshold_crossings(np.zeros(5), 1.0, direction="bogus")


class TestTriggerAverage:
    def test_basic_average(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            trigger_average,
        )

        # 같은 패턴이 100, 300, 500에 위치
        n_t = 1000
        signal = np.zeros((n_t,))
        pattern = np.linspace(-1.0, 1.0, 21)
        for trigger_pos in [100, 300, 500]:
            signal[trigger_pos - 10 : trigger_pos + 11] = pattern
        triggers = np.array([100, 300, 500])
        avg, count = trigger_average(signal, triggers, half_window=10)
        np.testing.assert_allclose(avg, pattern)
        assert count == 3

    def test_with_std(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            trigger_average,
        )

        rng = np.random.default_rng(0)
        signal = rng.standard_normal(1000)
        triggers = np.array([100, 200, 300, 400, 500])
        result = trigger_average(signal, triggers, half_window=20, return_std=True)
        avg, std, count = result
        assert avg.shape == (41,)
        assert std.shape == (41,)
        assert count == 5

    def test_trigger_signal_format(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            trigger_average,
        )

        n = 500
        signal = np.zeros(n)
        signal[100] = 5.0
        trig = np.zeros(n)
        trig[100] = 1.0
        avg, count = trigger_average(signal, trig, half_window=5)
        # 한 트리거만, 중심에서 5.0
        assert count == 1
        assert avg[5] == 5.0

    def test_invalid_half_window(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            trigger_average,
        )

        with pytest.raises(ValueError, match="half_window"):
            trigger_average(np.zeros(100), np.array([10]), half_window=0)

    def test_no_valid_triggers(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            trigger_average,
        )

        # 트리거가 너무 가장자리
        avg, count = trigger_average(np.zeros(100), np.array([2, 98]), half_window=10)
        assert count == 0
        np.testing.assert_array_equal(avg, np.zeros(21))

    def test_2d_signal(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            trigger_average,
        )

        rng = np.random.default_rng(1)
        signal = rng.standard_normal((1000, 5))
        triggers = np.array([200, 400, 600])
        avg, count = trigger_average(signal, triggers, half_window=20)
        assert avg.shape == (41, 5)


class TestConditionalAverage:
    def test_boolean_mask(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            conditional_average,
        )

        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cond = np.array([True, False, True, False, True])
        avg, n = conditional_average(signal, cond)
        np.testing.assert_allclose(avg, 3.0)
        assert n == 3

    def test_callable_condition(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            conditional_average,
        )

        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        avg, n = conditional_average(signal, lambda s: s > 2.5)
        np.testing.assert_allclose(avg, 4.0)  # mean of [3, 4, 5]
        assert n == 3

    def test_2d_signal(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            conditional_average,
        )

        signal = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        cond = np.array([True, False, True])
        avg, n = conditional_average(signal, cond)
        np.testing.assert_allclose(avg, [3.0, 4.0])
        assert n == 2

    def test_no_match(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            conditional_average,
        )

        signal = np.array([1.0, 2.0, 3.0])
        avg, n = conditional_average(signal, np.array([False, False, False]))
        assert n == 0
        assert avg == 0.0

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            conditional_average,
        )

        with pytest.raises(ValueError, match="condition"):
            conditional_average(np.zeros(5), np.array([True, False]))


class TestQuadrantMasks:
    def test_partition(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            quadrant_masks,
        )

        rng = np.random.default_rng(2)
        up = rng.standard_normal(1000)
        vp = rng.standard_normal(1000)
        Q1, Q2, Q3, Q4 = quadrant_masks(up, vp)
        # disjoint: 정확히 한 사분면에만 속함 (0 위치 제외)
        total = Q1.astype(int) + Q2.astype(int) + Q3.astype(int) + Q4.astype(int)
        # 0은 어느 사분면도 아님
        nonzero = (up != 0) & (vp != 0)
        np.testing.assert_array_equal(total[nonzero], np.ones(nonzero.sum()))

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            quadrant_masks,
        )

        with pytest.raises(ValueError, match="shape"):
            quadrant_masks(np.zeros(10), np.zeros(20))


class TestEventDuration:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            event_duration_stats,
        )

        cond = np.array([False, True, True, False, True, True, True, False])
        s = event_duration_stats(cond, dt=1.0)
        assert s["n_events"] == 2
        assert s["mean_duration"] == 2.5
        assert s["max_duration"] == 3.0
        assert s["total_active_time"] == 5.0
        assert s["active_fraction"] == 5 / 8

    def test_no_events(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            event_duration_stats,
        )

        s = event_duration_stats(np.zeros(10, dtype=bool))
        assert s["n_events"] == 0
        assert s["mean_duration"] == 0.0

    def test_all_active(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            event_duration_stats,
        )

        s = event_duration_stats(np.ones(20, dtype=bool), dt=0.1)
        assert s["n_events"] == 1
        assert abs(s["max_duration"] - 2.0) < 1e-12

    def test_empty(self) -> None:
        from naviertwin.core.flow_analysis.conditional_sampling import (
            event_duration_stats,
        )

        s = event_duration_stats(np.array([], dtype=bool))
        assert s["n_events"] == 0
