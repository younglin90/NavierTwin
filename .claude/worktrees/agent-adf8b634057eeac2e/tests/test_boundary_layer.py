"""Round 106 — 경계층 지표."""

from __future__ import annotations

import numpy as np


class TestBoundaryLayer:
    def test_delta99_tanh(self) -> None:
        from naviertwin.core.analysis.boundary_layer import delta99

        y = np.linspace(0, 2, 500)
        U = 10.0
        u = np.tanh(5 * y) * U
        d = delta99(y, u, U_edge=U)
        assert 0.4 < d < 0.7

    def test_displacement_momentum_thickness(self) -> None:
        from naviertwin.core.analysis.boundary_layer import (
            displacement_thickness,
            momentum_thickness,
        )

        # Linear profile u=U∞ · y/δ with δ=1 → δ* = 0.5, θ = 1/6
        delta = 1.0
        y = np.linspace(0, delta, 200)
        U = 1.0
        u = U * y / delta
        assert abs(displacement_thickness(y, u, U) - 0.5) < 1e-3
        assert abs(momentum_thickness(y, u, U) - 1 / 6) < 1e-3

    def test_wall_shear(self) -> None:
        from naviertwin.core.analysis.boundary_layer import (
            friction_velocity,
            wall_shear_stress,
        )

        # Poiseuille-like u(y) = A y → du/dy = A 상수
        A = 50.0
        y = np.linspace(0, 0.1, 100)
        u = A * y
        mu = 1e-3
        tau = wall_shear_stress(y, u, mu)
        assert abs(tau - mu * A) < 1e-6
        ut = friction_velocity(tau, rho=1000.0)
        assert ut > 0

    def test_tke(self) -> None:
        from naviertwin.core.analysis.boundary_layer import turbulent_ke

        rng = np.random.default_rng(0)
        up = rng.standard_normal(10000)
        vp = rng.standard_normal(10000)
        k = turbulent_ke(up, vp)
        # 2D → k ≈ 0.5 * (1+1) = 1
        assert 0.9 < k < 1.1
