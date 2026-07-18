"""Round 628 — unstructured cell volumes + volume integrals."""

from __future__ import annotations

import numpy as np
import pytest


class TestTetVolume:
    def test_unit_tet(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import tet_volume

        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        assert abs(tet_volume(v) - 1 / 6) < 1e-12

    def test_translated_invariant(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import tet_volume

        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) + 5.0
        assert abs(tet_volume(v) - 1 / 6) < 1e-12

    def test_invalid_shape_raises(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import tet_volume

        with pytest.raises(ValueError, match="4, 3"):
            tet_volume(np.zeros((3, 3)))

    def test_degenerate_zero(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import tet_volume

        # 4점이 같은 평면 → V = 0
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
        assert tet_volume(v) < 1e-12


class TestTetBatch:
    def test_two_tets(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import tet_volumes_batch

        # 두 단위 사면체 (분리)
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # tet 1
            [10, 10, 10], [11, 10, 10], [10, 11, 10], [10, 10, 11],  # tet 2
        ], dtype=float)
        conn = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        vols = tet_volumes_batch(verts, conn)
        np.testing.assert_allclose(vols, [1 / 6, 1 / 6], atol=1e-12)

    def test_invalid_vertex_shape_raises(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import tet_volumes_batch

        with pytest.raises(ValueError, match="vertices"):
            tet_volumes_batch(np.zeros((10, 2)), np.zeros((1, 4), dtype=int))

    def test_invalid_conn_shape_raises(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import tet_volumes_batch

        with pytest.raises(ValueError, match="connectivity"):
            tet_volumes_batch(np.zeros((10, 3)), np.zeros((1, 5), dtype=int))


class TestHexVolume:
    def test_unit_hex(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import hex_volume

        v = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)
        np.testing.assert_allclose(hex_volume(v), 1.0, atol=1e-12)

    def test_scaled(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import hex_volume

        v = np.array([
            [0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0],
            [0, 0, 5], [2, 0, 5], [2, 3, 5], [0, 3, 5],
        ], dtype=float)
        # V = 2 * 3 * 5 = 30
        np.testing.assert_allclose(hex_volume(v), 30.0, atol=1e-12)

    def test_invalid_shape(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import hex_volume

        with pytest.raises(ValueError, match="8, 3"):
            hex_volume(np.zeros((6, 3)))


class TestPyramidVolume:
    def test_unit_pyramid(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import pyramid_volume

        # 1×1 base, h=1 apex → V = 1/3
        v = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # base
            [0.5, 0.5, 1.0],  # apex
        ], dtype=float)
        np.testing.assert_allclose(pyramid_volume(v), 1 / 3, atol=1e-12)

    def test_invalid_shape(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import pyramid_volume

        with pytest.raises(ValueError, match="5, 3"):
            pyramid_volume(np.zeros((4, 3)))


class TestPrismVolume:
    def test_triangular_prism(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import prism_volume

        # 단위 직각 삼각 프리즘 (높이 1) → V = 0.5
        v = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],  # 아래
            [0, 0, 1], [1, 0, 1], [0, 1, 1],  # 위
        ], dtype=float)
        np.testing.assert_allclose(prism_volume(v), 0.5, atol=1e-12)

    def test_invalid_shape(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import prism_volume

        with pytest.raises(ValueError, match="6, 3"):
            prism_volume(np.zeros((5, 3)))


class TestVolumeIntegral:
    def test_constant_field(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import volume_integral

        V = np.array([1.0, 2.0, 3.0])
        f = np.ones(3) * 5.0
        # ∫ 5 dV = 5 * 6 = 30
        np.testing.assert_allclose(volume_integral(V, f), 30.0)

    def test_vector_field(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import volume_integral

        V = np.array([1.0, 2.0])
        f = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        result = volume_integral(V, f)
        np.testing.assert_allclose(result, [1.0, 2.0, 0.0])

    def test_shape_mismatch(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import volume_integral

        with pytest.raises(ValueError, match="length"):
            volume_integral(np.zeros(3), np.zeros(5))

    def test_volumes_2d_raises(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import volume_integral

        with pytest.raises(ValueError, match="1D"):
            volume_integral(np.zeros((3, 3)), np.zeros(3))


class TestVolumeAverage:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import volume_average

        V = np.array([1.0, 1.0, 1.0])
        f = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(volume_average(V, f), 2.0)

    def test_weighted_unequal(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import volume_average

        V = np.array([1.0, 3.0])
        f = np.array([1.0, 5.0])
        # (1*1 + 3*5)/(1+3) = 16/4 = 4
        np.testing.assert_allclose(volume_average(V, f), 4.0)

    def test_zero_volume(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import volume_average

        V = np.zeros(3)
        f = np.array([1.0, 2.0, 3.0])
        assert volume_average(V, f) == 0.0


class TestVolumeWeightedVariance:
    def test_constant_zero(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import (
            volume_weighted_variance,
        )

        V = np.array([1.0, 2.0, 3.0])
        f = np.array([5.0, 5.0, 5.0])
        np.testing.assert_allclose(volume_weighted_variance(V, f), 0.0)

    def test_known_value(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import (
            volume_weighted_variance,
        )

        V = np.array([1.0, 1.0])
        f = np.array([0.0, 4.0])
        # avg = 2, var = (1*4 + 1*4)/2 = 4
        np.testing.assert_allclose(volume_weighted_variance(V, f), 4.0)


class TestCellCentroids:
    def test_tet(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import cell_centroids

        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=float)
        conn = np.array([[0, 1, 2, 3]])
        c = cell_centroids(verts, conn)
        np.testing.assert_allclose(c, [[0.25, 0.25, 0.25]])

    def test_invalid_vertex_shape(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import cell_centroids

        with pytest.raises(ValueError, match="vertices"):
            cell_centroids(np.zeros((10, 2)), np.zeros((1, 4), dtype=int))

    def test_invalid_conn_shape(self) -> None:
        from naviertwin.core.flow_analysis.cell_volume import cell_centroids

        with pytest.raises(ValueError, match="2D"):
            cell_centroids(np.zeros((10, 3)), np.zeros(4, dtype=int))
