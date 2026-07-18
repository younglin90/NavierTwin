"""Round 101 — Taylor-Green vortex."""

from __future__ import annotations

import numpy as np
import pytest


class TestTaylorGreen:
    def test_divergence_free(self) -> None:
        from naviertwin.core.validation.taylor_green import taylor_green_2d

        n = 64
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        y = np.linspace(0, 2 * np.pi, n, endpoint=False)
        res = taylor_green_2d(x, y, t=0.0, nu=0.01)
        u, v = res["u"], res["v"]
        dx = x[1] - x[0]
        du_dx = np.gradient(u, dx, axis=1)
        dv_dy = np.gradient(v, dx, axis=0)
        div = du_dx + dv_dy
        assert np.max(np.abs(div)) < 1e-2

    def test_decay(self) -> None:
        from naviertwin.core.validation.taylor_green import (
            kinetic_energy_decay,
            taylor_green_2d,
        )

        x = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        y = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        nu = 0.05
        r0 = taylor_green_2d(x, y, t=0.0, nu=nu)
        r1 = taylor_green_2d(x, y, t=1.0, nu=nu)
        e0 = float(np.mean(r0["u"] ** 2 + r0["v"] ** 2))
        e1 = float(np.mean(r1["u"] ** 2 + r1["v"] ** 2))
        ratio = e1 / e0
        expected = kinetic_energy_decay(1.0, nu)
        assert ratio == pytest.approx(expected, rel=1e-6)
