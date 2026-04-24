"""Round 102 — RK4 streamline integration."""

from __future__ import annotations

import numpy as np


class TestStreamline:
    def test_rotational_field(self) -> None:
        from naviertwin.core.analysis.streamline import integrate_streamline

        # 회전 필드 (r, θ)에서 원을 그려야 함
        def vf(p):
            return np.array([-p[1], p[0]])

        path = integrate_streamline(
            vf, start=np.array([1.0, 0.0]), dt=0.01, n_steps=629,
        )
        # 끝점이 원점으로부터 ~1 거리
        r_end = np.linalg.norm(path[-1])
        assert abs(r_end - 1.0) < 1e-3
        # 2π 한 바퀴 후 거의 시작점 근처
        assert np.linalg.norm(path[-1] - path[0]) < 0.1

    def test_constant_field(self) -> None:
        from naviertwin.core.analysis.streamline import integrate_streamline

        def vf(p):  # noqa: ARG001
            return np.array([1.0, 0.0])

        path = integrate_streamline(
            vf, start=np.zeros(2), dt=0.1, n_steps=10,
        )
        assert abs(path[-1, 0] - 1.0) < 1e-12
        assert abs(path[-1, 1]) < 1e-12

    def test_multiple_seeds(self) -> None:
        from naviertwin.core.analysis.streamline import integrate_streamlines

        def vf(p):
            return np.array([p[0], -p[1]])  # saddle

        seeds = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        paths = integrate_streamlines(vf, seeds, dt=0.01, n_steps=50)
        assert paths.shape == (3, 51, 2)

    def test_stagnation(self) -> None:
        from naviertwin.core.analysis.streamline import integrate_streamline

        def vf(p):  # noqa: ARG001
            return np.array([0.0, 0.0])

        path = integrate_streamline(
            vf, start=np.array([2.0, 3.0]), dt=0.1, n_steps=100,
        )
        # 전구간 시작점 유지
        assert np.allclose(path, [2.0, 3.0])
