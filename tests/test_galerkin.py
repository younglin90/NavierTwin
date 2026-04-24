"""Round 126 — Galerkin ROM projection."""

from __future__ import annotations

import numpy as np


class TestGalerkin:
    def test_linear_projection_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.galerkin import (
            project_linear_operator,
        )

        Phi = np.eye(10)[:, :3]
        L = np.random.default_rng(0).standard_normal((10, 10))
        Lr = project_linear_operator(Phi, L)
        assert Lr.shape == (3, 3)

    def test_heat_rom_decay(self) -> None:
        """Laplacian 의 ROM RHS 로 열방정식 축소 동역학 적분 → 감쇠."""
        from naviertwin.core.analysis.time_integrator import integrate_ode
        from naviertwin.core.dimensionality_reduction.linear.galerkin import (
            project_field_to_modes,
            reconstruct_from_modes,
            rom_rhs_linear,
        )
        from naviertwin.core.linalg.sparse_builder import laplacian_1d

        n = 32
        L = np.asarray(laplacian_1d(n, h=1 / (n - 1)).toarray()) * 0.01  # ν=0.01
        # POD modes (sin 기저 근사)
        x = np.linspace(0, 1, n)
        Phi = np.stack([np.sin((k + 1) * np.pi * x) for k in range(3)], axis=1)
        # orthonormalize
        Phi, _ = np.linalg.qr(Phi)

        rhs = rom_rhs_linear(Phi, L, M=None)
        a0 = project_field_to_modes(Phi, np.sin(np.pi * x))
        _, ys = integrate_ode(rhs, a0, (0.0, 1.0), dt=0.01, method="rk4")
        # 계수 norm 감소
        assert np.linalg.norm(ys[-1]) < np.linalg.norm(ys[0])
        # 재구성
        u_end = reconstruct_from_modes(Phi, ys[-1])
        assert u_end.shape == (n,)

    def test_quadratic_projection(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.galerkin import (
            project_quadratic,
        )

        Phi = np.eye(5)[:, :2]

        def N(x, y):
            return x * y

        a = np.array([1.0, 2.0])
        r = project_quadratic(Phi, N, a)
        assert r.shape == (2,)

    def test_reconstruct_identity(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.galerkin import (
            project_field_to_modes,
            reconstruct_from_modes,
        )

        rng = np.random.default_rng(0)
        Phi, _ = np.linalg.qr(rng.standard_normal((20, 5)))
        x = Phi @ np.array([1.0, -2.0, 3.0, 0.5, -0.1])
        a = project_field_to_modes(Phi, x)
        x_re = reconstruct_from_modes(Phi, a)
        assert np.allclose(x, x_re, atol=1e-10)
