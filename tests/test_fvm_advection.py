"""Round 49 — FVM 이류 + 질량 보존."""

from __future__ import annotations

import numpy as np


class TestFVMUpwind:
    def test_advects_forward(self) -> None:
        from naviertwin.core.solver_interfaces.fvm_advection import fvm_upwind_1d

        N = 64
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        u0 = np.exp(-((x - np.pi) ** 2) / 0.2)
        t, U = fvm_upwind_1d(u0, c=1.0, L=2 * np.pi, T=np.pi, cfl=0.4)

        # 프로파일이 이동 (최대값 위치 shift)
        peak0 = int(np.argmax(u0))
        peakf = int(np.argmax(U[-1]))
        assert peakf != peak0
        assert np.all(np.isfinite(U))

    def test_mass_conservation(self) -> None:
        from naviertwin.core.solver_interfaces.fvm_advection import (
            fvm_upwind_1d,
            total_mass,
        )

        N = 32
        L = 2 * np.pi
        dx = L / N
        x = np.linspace(0, L, N, endpoint=False)
        u0 = np.sin(x) + 1.0  # 양의 질량
        m0 = total_mass(u0, dx)
        _, U = fvm_upwind_1d(u0, c=1.0, L=L, T=0.5, cfl=0.5)
        mf = total_mass(U[-1], dx)
        # 주기경계 upwind → 정확 보존
        assert abs(mf - m0) < 1e-10


class TestMUSCLHancock:
    def test_advects_smoothly(self) -> None:
        from naviertwin.core.solver_interfaces.fvm_advection import fvm_musclhancock_1d

        N = 64
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        u0 = np.exp(-((x - np.pi) ** 2) / 0.2)
        _, U = fvm_musclhancock_1d(u0, c=1.0, L=2 * np.pi, T=np.pi, cfl=0.3)
        # 일반적으로 upwind 보다 첨두 보존 잘 됨
        assert np.max(U[-1]) > 0.5 * np.max(u0)


class TestMinmod:
    def test_positive_pair(self) -> None:
        from naviertwin.core.solver_interfaces.fvm_advection import minmod

        a = np.array([1.0, 2.0])
        b = np.array([3.0, 1.5])
        r = minmod(a, b)
        assert np.allclose(r, [1.0, 1.5])

    def test_opposite_sign(self) -> None:
        from naviertwin.core.solver_interfaces.fvm_advection import minmod

        r = minmod(np.array([1.0]), np.array([-1.0]))
        assert r[0] == 0.0
