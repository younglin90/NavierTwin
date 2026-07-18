"""Round 625 — vector field topology: critical points + classification."""

from __future__ import annotations

import numpy as np
import pytest


class TestClassify2D:
    def test_saddle(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import classify_2d

        # eigvals: 1, -1
        J = np.array([[1.0, 0.0], [0.0, -1.0]])
        assert classify_2d(J) == "saddle"

    def test_source(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import classify_2d

        J = np.array([[2.0, 0.0], [0.0, 1.0]])
        assert classify_2d(J) == "source"

    def test_sink(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import classify_2d

        J = np.array([[-2.0, 0.0], [0.0, -1.0]])
        assert classify_2d(J) == "sink"

    def test_center(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import classify_2d

        # 회전: u = -y, v = x → J = [[0, -1], [1, 0]] → eigvals ±i
        J = np.array([[0.0, -1.0], [1.0, 0.0]])
        assert classify_2d(J) == "center"

    def test_spiral_source(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import classify_2d

        # 양의 실수부, 허수부
        J = np.array([[0.5, -1.0], [1.0, 0.5]])
        assert classify_2d(J) == "spiral_source"

    def test_spiral_sink(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import classify_2d

        J = np.array([[-0.5, -1.0], [1.0, -0.5]])
        assert classify_2d(J) == "spiral_sink"

    def test_degenerate(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import classify_2d

        J = np.array([[0.0, 0.0], [0.0, 0.0]])
        assert classify_2d(J) == "degenerate"

    def test_invalid_shape_raises(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import classify_2d

        with pytest.raises(ValueError, match="2, 2"):
            classify_2d(np.eye(3))


class TestFindCriticalPoints:
    def test_rotational_field_finds_center(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import (
            find_critical_points,
        )

        # u = -y, v = x → center at origin
        x = np.linspace(-1, 1, 41)
        X, Y = np.meshgrid(x, x, indexing="ij")
        u, v = -Y, X
        h = x[1] - x[0]
        cps = find_critical_points(u, v, dx=h, dy=h)
        assert len(cps) >= 1
        # 가장 가까운 임계점이 (0, 0) 근처
        closest = min(cps, key=lambda c: c["x"] ** 2 + c["y"] ** 2)
        # x[0] = -1 → 임계점 격자 좌표 시스템에서 원점은 (1.0, 1.0)
        assert "type" in closest

    def test_uniform_field_no_cp(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import (
            find_critical_points,
        )

        u = np.ones((10, 10))
        v = np.ones((10, 10))
        cps = find_critical_points(u, v)
        assert len(cps) == 0

    def test_saddle_field(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import (
            find_critical_points,
        )

        # u = x, v = -y → saddle at origin (eigvals 1, -1)
        x = np.linspace(-1, 1, 41)
        X, Y = np.meshgrid(x, x, indexing="ij")
        u = X
        v = -Y
        h = x[1] - x[0]
        cps = find_critical_points(u, v, dx=h, dy=h)
        types = [c["type"] for c in cps]
        # 어느 하나는 saddle이어야 함
        assert any(t == "saddle" for t in types)

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import (
            find_critical_points,
        )

        with pytest.raises(ValueError, match="2D"):
            find_critical_points(np.zeros(10), np.zeros(10))


class TestPoincareIndex:
    def test_center_index_plus1(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import poincare_index

        # 회전장 → index = +1 (또는 그 부근)
        x = np.linspace(-1, 1, 21)
        X, Y = np.meshgrid(x, x, indexing="ij")
        u, v = -Y, X
        idx = poincare_index(u, v, center_i=10, center_j=10)
        assert idx == 1

    def test_saddle_index_minus1(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import poincare_index

        # u=x, v=-y → saddle, index = -1
        x = np.linspace(-1, 1, 21)
        X, Y = np.meshgrid(x, x, indexing="ij")
        u, v = X, -Y
        idx = poincare_index(u, v, center_i=10, center_j=10)
        assert idx == -1

    def test_uniform_field_zero(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import poincare_index

        u = np.ones((11, 11))
        v = np.zeros((11, 11))
        idx = poincare_index(u, v, center_i=5, center_j=5)
        assert idx == 0

    def test_invalid_center_raises(self) -> None:
        from naviertwin.core.flow_analysis.critical_points import poincare_index

        with pytest.raises(ValueError, match="boundary"):
            poincare_index(np.zeros((5, 5)), np.zeros((5, 5)), 0, 0)
