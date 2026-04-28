"""Round 615 — slice + line probe extraction."""

from __future__ import annotations

import numpy as np
import pytest


class TestSliceAxisAligned:
    def test_basic_3d(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import slice_axis_aligned

        f = np.arange(60).reshape(3, 4, 5).astype(float)
        sl = slice_axis_aligned(f, axis=2, position=2)
        assert sl.shape == (3, 4)
        # f[i, j, 2]
        np.testing.assert_array_equal(sl, f[:, :, 2])

    def test_axis_0(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import slice_axis_aligned

        f = np.arange(24).reshape(2, 3, 4).astype(float)
        sl = slice_axis_aligned(f, axis=0, position=1)
        assert sl.shape == (3, 4)

    def test_invalid_axis_raises(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import slice_axis_aligned

        with pytest.raises(ValueError, match="axis"):
            slice_axis_aligned(np.zeros((2, 3, 4)), axis=5, position=0)

    def test_invalid_position_raises(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import slice_axis_aligned

        with pytest.raises(ValueError, match="position"):
            slice_axis_aligned(np.zeros((2, 3, 4)), axis=2, position=10)


class TestSlicePlaneArbitrary:
    def test_z_zero_plane(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import slice_plane_arbitrary

        rng = np.random.default_rng(0)
        n = 1000
        pts = rng.uniform(-1, 1, (n, 3))
        f = pts[:, 2] ** 2
        sel_pts, sel_f = slice_plane_arbitrary(
            pts, f,
            plane_origin=np.array([0.0, 0.0, 0.0]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
            tolerance=0.05,
        )
        assert len(sel_pts) > 0
        assert np.all(np.abs(sel_pts[:, 2]) < 0.05)

    def test_no_points_in_tolerance(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import slice_plane_arbitrary

        pts = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
        f = np.array([1.0, 2.0])
        sel_pts, sel_f = slice_plane_arbitrary(
            pts, f,
            plane_origin=np.array([0.0, 0.0, 0.0]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
            tolerance=0.001,
        )
        assert len(sel_pts) == 0

    def test_invalid_points_shape_raises(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import slice_plane_arbitrary

        with pytest.raises(ValueError, match="points"):
            slice_plane_arbitrary(
                np.zeros((5, 2)), np.zeros(5),
                np.zeros(3), np.array([1.0, 0, 0]),
            )

    def test_invalid_plane_shape_raises(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import slice_plane_arbitrary

        with pytest.raises(ValueError, match="plane"):
            slice_plane_arbitrary(
                np.zeros((5, 3)), np.zeros(5),
                np.zeros(2), np.zeros(3),
            )

    def test_field_length_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import slice_plane_arbitrary

        with pytest.raises(ValueError, match="field length"):
            slice_plane_arbitrary(
                np.zeros((5, 3)), np.zeros(3),
                np.zeros(3), np.array([1.0, 0, 0]),
            )


class TestLineProbe:
    def test_nearest_method(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import line_probe

        pts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        f = np.array([10.0, 20.0, 30.0, 40.0])
        line_pts, sampled = line_probe(
            pts, f,
            start=np.array([0.0, 0.0, 0.0]),
            end=np.array([3.0, 0.0, 0.0]),
            n_samples=4,
            method="nearest",
        )
        assert line_pts.shape == (4, 3)
        assert sampled.shape == (4,)
        np.testing.assert_allclose(sampled, [10.0, 20.0, 30.0, 40.0])

    def test_idw_smooth(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import line_probe

        rng = np.random.default_rng(1)
        pts = rng.uniform(-1, 1, (50, 3))
        f = pts[:, 0]  # x-linear field
        line_pts, sampled = line_probe(
            pts, f,
            start=np.array([-0.5, 0.0, 0.0]),
            end=np.array([0.5, 0.0, 0.0]),
            n_samples=10,
            method="idw",
        )
        # 단조 증가
        assert np.all(np.diff(sampled) >= -0.2)

    def test_invalid_method_raises(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import line_probe

        with pytest.raises(ValueError, match="method"):
            line_probe(
                np.zeros((5, 3)), np.zeros(5),
                np.zeros(3), np.ones(3), method="bogus",
            )

    def test_too_few_samples_raises(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import line_probe

        with pytest.raises(ValueError, match="n_samples"):
            line_probe(
                np.zeros((5, 3)), np.zeros(5),
                np.zeros(3), np.ones(3), n_samples=1,
            )

    def test_invalid_pts_shape_raises(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import line_probe

        with pytest.raises(ValueError, match="points"):
            line_probe(
                np.zeros((5, 2)), np.zeros(5),
                np.zeros(3), np.ones(3),
            )

    def test_invalid_endpoints_raises(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import line_probe

        with pytest.raises(ValueError, match="start/end"):
            line_probe(
                np.zeros((5, 3)), np.zeros(5),
                np.zeros(2), np.ones(3),
            )


class TestArcLength:
    def test_straight_line(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import polyline_arc_length

        pts = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
        s = polyline_arc_length(pts)
        np.testing.assert_allclose(s, [0.0, 1.0, 2.0])

    def test_lshape(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import polyline_arc_length

        pts = np.array([[0.0, 0], [1.0, 0], [1.0, 1.0]])
        s = polyline_arc_length(pts)
        np.testing.assert_allclose(s, [0.0, 1.0, 2.0])

    def test_2d_required(self) -> None:
        from naviertwin.core.flow_analysis.slice_extract import polyline_arc_length

        with pytest.raises(ValueError, match="2D"):
            polyline_arc_length(np.zeros(5))
