"""Round 614 — coordinate transformations + axis-align rotation."""

from __future__ import annotations

import numpy as np
import pytest


class TestCartCylRoundTrip:
    def test_round_trip(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import (
            cart_to_cyl,
            cyl_to_cart,
        )

        rng = np.random.default_rng(0)
        xyz = rng.standard_normal((50, 3))
        rtz = cart_to_cyl(xyz)
        xyz_back = cyl_to_cart(rtz)
        np.testing.assert_allclose(xyz_back, xyz, atol=1e-12)

    def test_x_axis(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import cart_to_cyl

        xyz = np.array([[1.0, 0.0, 0.0]])
        rtz = cart_to_cyl(xyz)
        np.testing.assert_allclose(rtz, [[1.0, 0.0, 0.0]], atol=1e-12)

    def test_y_axis(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import cart_to_cyl

        xyz = np.array([[0.0, 1.0, 0.0]])
        rtz = cart_to_cyl(xyz)
        np.testing.assert_allclose(rtz, [[1.0, np.pi / 2, 0.0]], atol=1e-12)

    def test_invalid_shape_raises(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import cart_to_cyl

        with pytest.raises(ValueError, match="N, 3"):
            cart_to_cyl(np.zeros((5, 2)))

    def test_cyl_to_cart_invalid_shape_raises(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import cyl_to_cart

        with pytest.raises(ValueError, match="N, 3"):
            cyl_to_cart(np.zeros((5, 4)))


class TestCartSph:
    def test_round_trip(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import (
            cart_to_sph,
            sph_to_cart,
        )

        # 양의 영역 (atan2 + arccos가 unique)
        rng = np.random.default_rng(1)
        xyz = np.abs(rng.standard_normal((30, 3))) + 0.1
        rtp = cart_to_sph(xyz)
        xyz_back = sph_to_cart(rtp)
        np.testing.assert_allclose(xyz_back, xyz, atol=1e-12)

    def test_z_axis(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import cart_to_sph

        xyz = np.array([[0.0, 0.0, 1.0]])
        rtp = cart_to_sph(xyz)
        np.testing.assert_allclose(rtp[:, 0], [1.0])
        # θ = 0 (북극)
        np.testing.assert_allclose(rtp[:, 1], [0.0], atol=1e-12)

    def test_origin(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import cart_to_sph

        xyz = np.array([[0.0, 0.0, 0.0]])
        rtp = cart_to_sph(xyz)
        np.testing.assert_allclose(rtp, [[0.0, 0.0, 0.0]], atol=1e-12)

    def test_invalid_shape_raises(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import (
            cart_to_sph,
            sph_to_cart,
        )

        with pytest.raises(ValueError, match="N, 3"):
            cart_to_sph(np.zeros((5,)))
        with pytest.raises(ValueError, match="N, 3"):
            sph_to_cart(np.zeros((5, 5)))


class TestVectorCartCyl:
    def test_radial_to_cyl(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import vector_cart_to_cyl

        # 위치 +x 축, 벡터 +x → 원통계에서 v_r = 1, v_θ = 0
        pos = np.array([[1.0, 0.0, 0.0]])
        vec = np.array([[1.0, 0.0, 0.0]])
        v_cyl = vector_cart_to_cyl(vec, pos)
        np.testing.assert_allclose(v_cyl, [[1.0, 0.0, 0.0]], atol=1e-12)

    def test_tangential_at_y_axis(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import vector_cart_to_cyl

        # 위치 +y 축, 벡터 +x → v_r = 0 (직각), v_θ = -1 (시계방향)
        pos = np.array([[0.0, 1.0, 0.0]])
        vec = np.array([[1.0, 0.0, 0.0]])
        v_cyl = vector_cart_to_cyl(vec, pos)
        np.testing.assert_allclose(v_cyl, [[0.0, -1.0, 0.0]], atol=1e-12)

    def test_round_trip(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import (
            vector_cart_to_cyl,
            vector_cyl_to_cart,
        )

        rng = np.random.default_rng(2)
        pos = rng.standard_normal((20, 3))
        vec = rng.standard_normal((20, 3))
        v_cyl = vector_cart_to_cyl(vec, pos)
        v_back = vector_cyl_to_cart(v_cyl, pos)
        np.testing.assert_allclose(v_back, vec, atol=1e-12)

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import vector_cart_to_cyl

        with pytest.raises(ValueError, match="mismatch"):
            vector_cart_to_cyl(np.zeros((5, 3)), np.zeros((4, 3)))

    def test_wrong_dim_raises(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import vector_cart_to_cyl

        with pytest.raises(ValueError, match="N, 3"):
            vector_cart_to_cyl(np.zeros((5, 2)), np.zeros((5, 2)))


class TestAxisAlign:
    def test_already_aligned(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import axis_align_rotation

        R = axis_align_rotation(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_z_to_x(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import axis_align_rotation

        R = axis_align_rotation(
            np.array([0.0, 0.0, 1.0]),
            target=np.array([1.0, 0.0, 0.0]),
        )
        z = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(R @ z, [1.0, 0.0, 0.0], atol=1e-12)

    def test_anti_parallel(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import axis_align_rotation

        # +z → -z (180도 회전)
        R = axis_align_rotation(
            np.array([0.0, 0.0, 1.0]),
            target=np.array([0.0, 0.0, -1.0]),
        )
        z = np.array([0.0, 0.0, 1.0])
        out = R @ z
        np.testing.assert_allclose(out, [0.0, 0.0, -1.0], atol=1e-12)

    def test_invalid_shape_raises(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import axis_align_rotation

        with pytest.raises(ValueError, match="3,"):
            axis_align_rotation(np.zeros(2))

    def test_arbitrary_axis(self) -> None:
        from naviertwin.core.flow_analysis.coord_transform import axis_align_rotation

        a = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        R = axis_align_rotation(a)
        out = R @ a
        np.testing.assert_allclose(out, [0.0, 0.0, 1.0], atol=1e-12)
