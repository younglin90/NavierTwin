"""Round 46 — SPH kernel + density + gradient."""

from __future__ import annotations

import numpy as np
import pytest


class TestCubicSplineKernel:
    def test_center_value(self) -> None:
        from naviertwin.core.solver_interfaces.sph import cubic_spline_kernel

        w = cubic_spline_kernel(np.array([0.0]), h=1.0, dim=1)
        assert abs(float(w[0]) - 2 / 3) < 1e-10

    def test_zero_outside_support(self) -> None:
        from naviertwin.core.solver_interfaces.sph import cubic_spline_kernel

        w = cubic_spline_kernel(np.array([2.1, 3.0, -3.0]), h=1.0, dim=1)
        assert np.all(w == 0.0)

    def test_invalid_dim(self) -> None:
        from naviertwin.core.solver_interfaces.sph import cubic_spline_kernel

        with pytest.raises(ValueError):
            cubic_spline_kernel(np.array([0.0]), h=1.0, dim=4)


class TestSPHDensity:
    def test_uniform_particles(self) -> None:
        from naviertwin.core.solver_interfaces.sph import sph_density_1d

        # 균일 분포 → 균일 밀도
        x = np.arange(20, dtype=float) * 0.5
        m = np.ones(20) * 0.5
        rho = sph_density_1d(x, m, h=1.0)
        # 경계 제외 내부는 거의 균일
        inner = rho[5:-5]
        assert float(inner.std()) / float(inner.mean()) < 0.05


class TestSPHGradient:
    def test_constant_field_zero_gradient(self) -> None:
        from naviertwin.core.solver_interfaces.sph import sph_gradient_1d

        x = np.linspace(0, 5, 30)
        v = np.ones(30) * 3.0
        m = np.ones(30) * 0.2
        grad = sph_gradient_1d(x, v, m, h=1.0)
        # 상수 장 → 모든 v_j - v_i = 0 → gradient ≈ 0
        assert np.all(np.abs(grad) < 1e-10)

    def test_linear_field_gradient_magnitude(self) -> None:
        from naviertwin.core.solver_interfaces.sph import sph_gradient_1d

        x = np.linspace(0, 5, 40)
        v = 2.0 * x  # 선형, 기울기 2
        m = np.ones(40) * 0.15
        grad = sph_gradient_1d(x, v, m, h=1.0)
        # 내부에서 gradient magnitude 는 0 보다 큰 상수 근사 (부호는 커널 방향 의존)
        inner = grad[10:-10]
        assert float(np.std(inner)) < 0.1  # 내부 플래토
        assert float(np.abs(inner.mean())) > 1.0  # 비트리비얼 크기
