"""Round 122 — 2D 이류-확산."""

from __future__ import annotations

import numpy as np


class TestConvDiff2D:
    def test_gaussian_advects(self) -> None:
        """가우시안 범프가 (u0, v0) 방향으로 이동."""
        from naviertwin.core.solvers.conv_diff_2d import solve_conv_diff_2d

        x, y, t, C = solve_conv_diff_2d(
            nx=48, ny=48, T=0.2,
            u0=1.0, v0=0.0, D=1e-3,
        )
        # 초기 최대값 위치 → 이동 후 x 방향으로 이동
        i0, _ = np.unravel_index(np.argmax(C[:, :, 0]), C[:, :, 0].shape)
        i1, _ = np.unravel_index(np.argmax(C[:, :, -1]), C[:, :, -1].shape)
        assert i1 > i0

    def test_pure_diffusion_preserves_mass(self) -> None:
        """순수 확산 (u=v=0) → 총량 보존 (경계 영향 무시 → 범프가 boundary 에 닿기 전)."""
        from naviertwin.core.solvers.conv_diff_2d import solve_conv_diff_2d

        x, y, t, C = solve_conv_diff_2d(
            nx=64, ny=64, T=0.02,
            u0=0.0, v0=0.0, D=0.005,
        )
        m0 = float(C[:, :, 0].sum())
        m1 = float(C[:, :, -1].sum())
        assert abs(m1 - m0) / m0 < 0.01
