"""Round 613 — snapshot time interpolation + sliding window."""

from __future__ import annotations

import numpy as np
import pytest


class TestInterpField:
    def test_linear_basic(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import interp_field

        snaps = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        times = np.array([0.0, 1.0, 2.0])
        u = interp_field(snaps, times, t_query=0.5)
        np.testing.assert_allclose(u, [2.0, 3.0])

    def test_linear_array_query(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import interp_field

        snaps = np.array([[0.0], [1.0], [2.0]])
        times = np.array([0.0, 1.0, 2.0])
        u = interp_field(snaps, times, t_query=np.array([0.25, 0.75, 1.5]))
        np.testing.assert_allclose(u[:, 0], [0.25, 0.75, 1.5])

    def test_clamp_below_range(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import interp_field

        snaps = np.array([[1.0], [2.0]])
        times = np.array([0.0, 1.0])
        u = interp_field(snaps, times, t_query=-1.0)
        np.testing.assert_allclose(u, [1.0])

    def test_clamp_above_range(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import interp_field

        snaps = np.array([[1.0], [2.0]])
        times = np.array([0.0, 1.0])
        u = interp_field(snaps, times, t_query=5.0)
        np.testing.assert_allclose(u, [2.0])

    def test_cubic_smooth(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import interp_field

        # 3차 다항식이라 cubic spline이 정확 재현
        times = np.linspace(0, 1, 10)
        snaps = times[:, None] ** 3
        u = interp_field(snaps, times, t_query=0.45, method="cubic")
        np.testing.assert_allclose(u, [0.45 ** 3], atol=1e-3)

    def test_cubic_falls_back_for_short_data(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import interp_field

        snaps = np.array([[1.0], [2.0], [3.0]])
        times = np.array([0.0, 1.0, 2.0])
        u = interp_field(snaps, times, t_query=0.5, method="cubic")
        # 3개 점 → linear fallback
        np.testing.assert_allclose(u, [1.5])

    def test_nearest(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import interp_field

        snaps = np.array([[10.0], [20.0]])
        times = np.array([0.0, 1.0])
        u = interp_field(snaps, times, t_query=0.3, method="nearest")
        np.testing.assert_allclose(u, [10.0])
        u = interp_field(snaps, times, t_query=0.7, method="nearest")
        np.testing.assert_allclose(u, [20.0])

    def test_invalid_method_raises(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import interp_field

        with pytest.raises(ValueError, match="method"):
            interp_field(
                np.zeros((3, 2)), np.array([0.0, 1.0, 2.0]),
                t_query=0.5, method="bogus",
            )

    def test_non_monotonic_times_raises(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import interp_field

        with pytest.raises(ValueError, match="increasing"):
            interp_field(
                np.zeros((3, 2)), np.array([0.0, 0.5, 0.3]), t_query=0.4,
            )

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import interp_field

        with pytest.raises(ValueError, match="mismatch"):
            interp_field(
                np.zeros((3, 2)), np.array([0.0, 1.0]), t_query=0.5,
            )

    def test_times_2d_raises(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import interp_field

        with pytest.raises(ValueError, match="1D"):
            interp_field(
                np.zeros((2, 2)),
                np.array([[0.0, 1.0], [0.0, 1.0]]),
                t_query=0.5,
            )

    def test_higher_dim_field(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import interp_field

        # (n_t, n_x, n_y, 3) — 3D 벡터 필드
        snaps = np.random.default_rng(0).standard_normal((4, 5, 5, 3))
        times = np.array([0.0, 1.0, 2.0, 3.0])
        u = interp_field(snaps, times, t_query=1.5)
        assert u.shape == (5, 5, 3)


class TestResampleUniform:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import resample_uniform

        snaps = np.array([[0.0], [1.0], [4.0], [9.0]])
        times = np.array([0.0, 1.0, 2.0, 3.0])
        t_u, u = resample_uniform(snaps, times, n_uniform=7)
        assert len(t_u) == 7
        assert u.shape == (7, 1)
        # 균일 간격
        np.testing.assert_allclose(np.diff(t_u), np.diff(t_u)[0])


class TestTimeAverageWindow:
    def test_window_average(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import time_average_window

        snaps = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        u = time_average_window(snaps, times, t_center=2.0, half_width=1.0)
        # t in [1, 3] → snaps[1:4] → mean = 3
        np.testing.assert_allclose(u, [3.0])

    def test_empty_window_returns_zero(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import time_average_window

        snaps = np.array([[1.0], [2.0]])
        times = np.array([0.0, 10.0])
        u = time_average_window(snaps, times, t_center=5.0, half_width=0.5)
        np.testing.assert_allclose(u, [0.0])

    def test_invalid_half_width_raises(self) -> None:
        from naviertwin.core.flow_analysis.time_interp import time_average_window

        with pytest.raises(ValueError, match="half_width"):
            time_average_window(
                np.zeros((5, 1)), np.linspace(0, 1, 5),
                t_center=0.5, half_width=0.0,
            )
