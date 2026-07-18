"""Round 104 — vorticity / Q-criterion."""

from __future__ import annotations

import numpy as np


class TestVorticity:
    def test_solid_body_rotation(self) -> None:
        """u=-Ωy, v=Ωx → ωz = 2Ω."""
        from naviertwin.core.analysis.vorticity import vorticity_2d

        n = 32
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y, indexing="xy")
        Omega = 2.5
        u = -Omega * Y
        v = Omega * X
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        w = vorticity_2d(u, v, dx, dy)
        # 내부 영역 평균 ≈ 2Ω
        center = w[5:-5, 5:-5]
        assert abs(center.mean() - 2 * Omega) < 1e-8

    def test_q_sign(self) -> None:
        """회전장 (solid rotation) → Q > 0 (vortex 탐지)."""
        from naviertwin.core.analysis.vorticity import q_criterion_2d

        n = 32
        x = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, x, indexing="xy")
        u = -Y
        v = X
        Q = q_criterion_2d(u, v, x[1] - x[0], x[1] - x[0])
        assert Q[5:-5, 5:-5].mean() > 0

    def test_vorticity_3d_shape(self) -> None:
        from naviertwin.core.analysis.vorticity import vorticity_3d

        rng = np.random.default_rng(0)
        u = rng.standard_normal((8, 10, 12))
        v = rng.standard_normal((8, 10, 12))
        w = rng.standard_normal((8, 10, 12))
        wx, wy, wz = vorticity_3d(u, v, w)
        assert wx.shape == u.shape
        assert wy.shape == u.shape
        assert wz.shape == u.shape
