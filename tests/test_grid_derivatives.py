"""Round 624 — uniform-grid finite difference (2nd/4th order)."""

from __future__ import annotations

import numpy as np
import pytest


class TestGradient2D:
    def test_quadratic_field(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import gradient_2d

        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = X ** 2 + Y ** 2
        h = x[1] - x[0]
        gx, gy = gradient_2d(f, dx=h, dy=h, order=4)
        # 중앙: ∂f/∂x = 2x, ∂f/∂y = 2y
        np.testing.assert_allclose(gx[10:-10, 10:-10], 2 * X[10:-10, 10:-10], atol=1e-3)
        np.testing.assert_allclose(gy[10:-10, 10:-10], 2 * Y[10:-10, 10:-10], atol=1e-3)

    def test_order_2_works(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import gradient_2d

        x = np.linspace(0, 1, 30)
        f = np.outer(x ** 2, np.ones(30))
        h = x[1] - x[0]
        gx, gy = gradient_2d(f, dx=h, dy=h, order=2)
        # 양 끝 제외; broadcast (n,) over (n, 30)
        expected = (2 * x[5:-5])[:, None] * np.ones((1, 20))
        np.testing.assert_allclose(gx[5:-5, 5:-5], expected, atol=1e-1)

    def test_invalid_order_raises(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import gradient_2d

        with pytest.raises(ValueError, match="order"):
            gradient_2d(np.zeros((10, 10)), order=3)

    def test_invalid_h_raises(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import gradient_2d

        with pytest.raises(ValueError, match="h"):
            gradient_2d(np.zeros((10, 10)), dx=0.0)

    def test_invalid_ndim_raises(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import gradient_2d

        with pytest.raises(ValueError, match="2D"):
            gradient_2d(np.zeros(10))

    def test_short_axis_falls_back(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import gradient_2d

        # 4점인 축이 있을 때 4차 → 2차로 폴백 (raise 안 함)
        f = np.zeros((4, 10))
        gx, gy = gradient_2d(f, order=4)
        assert gx.shape == f.shape


class TestGradient3D:
    def test_basic_shape(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import gradient_3d

        f = np.zeros((10, 10, 10))
        gx, gy, gz = gradient_3d(f)
        assert gx.shape == gy.shape == gz.shape == f.shape

    def test_invalid_ndim_raises(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import gradient_3d

        with pytest.raises(ValueError, match="3D"):
            gradient_3d(np.zeros((10, 10)))


class TestDivergence3D:
    def test_uniform_field_zero(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import divergence_3d

        u = np.ones((10, 10, 10))
        v = np.ones((10, 10, 10))
        w = np.ones((10, 10, 10))
        d = divergence_3d(u, v, w)
        np.testing.assert_allclose(d, 0.0, atol=1e-12)

    def test_linear_field(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import divergence_3d

        # u = x, v = y, w = z → div = 3
        x = np.linspace(0, 1, 30)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        h = x[1] - x[0]
        d = divergence_3d(X, Y, Z, dx=h, dy=h, dz=h, order=4)
        # 중앙
        np.testing.assert_allclose(d[5:-5, 5:-5, 5:-5], 3.0, atol=1e-3)

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import divergence_3d

        with pytest.raises(ValueError, match="same-shape"):
            divergence_3d(
                np.zeros((10, 10, 10)),
                np.zeros((9, 10, 10)),
                np.zeros((10, 10, 10)),
            )


class TestCurl3D:
    def test_uniform_zero(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import curl_3d

        u = np.ones((10, 10, 10))
        wx, wy, wz = curl_3d(u, u, u)
        np.testing.assert_allclose(wx, 0.0, atol=1e-12)
        np.testing.assert_allclose(wy, 0.0, atol=1e-12)
        np.testing.assert_allclose(wz, 0.0, atol=1e-12)

    def test_solid_body_rotation_z(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import curl_3d

        # u = -y, v = x, w = 0 → curl = (0, 0, 2)
        x = np.linspace(-1, 1, 30)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        u = -Y
        v = X
        w = np.zeros_like(X)
        h = x[1] - x[0]
        wx, wy, wz = curl_3d(u, v, w, dx=h, dy=h, dz=h, order=4)
        np.testing.assert_allclose(wz[5:-5, 5:-5, 5:-5], 2.0, atol=1e-3)
        np.testing.assert_allclose(wx[5:-5, 5:-5, 5:-5], 0.0, atol=1e-3)

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import curl_3d

        with pytest.raises(ValueError, match="same-shape"):
            curl_3d(
                np.zeros((10, 10)),
                np.zeros((10, 10)),
                np.zeros((10, 10)),
            )


class TestLaplacian:
    def test_2d_quadratic(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import laplacian_2d

        # f = x² + y² → ∇²f = 4
        x = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, x, indexing="ij")
        f = X ** 2 + Y ** 2
        h = x[1] - x[0]
        L = laplacian_2d(f, dx=h, dy=h, order=4)
        # 중앙
        np.testing.assert_allclose(L[10:-10, 10:-10], 4.0, atol=0.05)

    def test_3d_quadratic(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import laplacian_3d

        # f = x² + y² + z² → ∇²f = 6
        x = np.linspace(0, 1, 30)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        f = X ** 2 + Y ** 2 + Z ** 2
        h = x[1] - x[0]
        L = laplacian_3d(f, dx=h, dy=h, dz=h, order=4)
        np.testing.assert_allclose(L[5:-5, 5:-5, 5:-5], 6.0, atol=0.1)

    def test_2d_invalid_ndim_raises(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import laplacian_2d

        with pytest.raises(ValueError, match="2D"):
            laplacian_2d(np.zeros(10))

    def test_3d_invalid_ndim_raises(self) -> None:
        from naviertwin.core.flow_analysis.grid_derivatives import laplacian_3d

        with pytest.raises(ValueError, match="3D"):
            laplacian_3d(np.zeros((10, 10)))
