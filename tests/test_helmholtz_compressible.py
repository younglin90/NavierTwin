"""Round 37 — Helmholtz decomposition + compressible utils."""

from __future__ import annotations

import numpy as np
import pytest


class TestHelmholtz:
    def test_reconstruction(self) -> None:
        from naviertwin.core.flow_analysis.helmholtz import helmholtz_2d

        rng = np.random.default_rng(0)
        u = rng.standard_normal((16, 16))
        v = rng.standard_normal((16, 16))
        u_s, v_s, u_i, v_i = helmholtz_2d(u, v)
        assert np.allclose(u, u_s + u_i, atol=1e-10)
        assert np.allclose(v, v_s + v_i, atol=1e-10)

    def test_solenoidal_reduces_divergence(self) -> None:
        """원본 보다 solenoidal 부분의 스펙트럴 divergence 가 작아야."""
        from naviertwin.core.flow_analysis.helmholtz import helmholtz_2d

        rng = np.random.default_rng(0)
        N = 32
        u = rng.standard_normal((N, N))
        v = rng.standard_normal((N, N))
        u_s, v_s, _, _ = helmholtz_2d(u, v)

        # 같은 스펙트럴 차분으로 원본/solenoidal 비교
        def _div(ux: np.ndarray, vy: np.ndarray) -> float:
            U = np.fft.fft2(ux)
            V = np.fft.fft2(vy)
            kx = np.fft.fftfreq(N, d=2 * np.pi / N) * 2 * np.pi
            KX, KY = np.meshgrid(kx, kx)
            DIV_hat = 1j * (KX * U + KY * V)
            return float(np.abs(np.real(np.fft.ifft2(DIV_hat))).max())

        assert _div(u_s, v_s) <= _div(u, v)

    def test_shape_mismatch(self) -> None:
        from naviertwin.core.flow_analysis.helmholtz import helmholtz_2d

        with pytest.raises(ValueError):
            helmholtz_2d(np.zeros((5, 5)), np.zeros((5, 6)))


class TestCompressible:
    def test_speed_of_sound_air(self) -> None:
        from naviertwin.core.flow_analysis.thermofluids.compressible import (
            speed_of_sound,
        )

        a = float(speed_of_sound(gamma=1.4, R=287, T=300))
        assert abs(a - 347.2) < 0.5

    def test_isentropic_ratios_sonic(self) -> None:
        """M=1 에서 표준 값: p_0/p = 1.893, T_0/T = 1.2, ρ_0/ρ = 1.577."""
        from naviertwin.core.flow_analysis.thermofluids.compressible import (
            isentropic_p_ratio,
            isentropic_rho_ratio,
            isentropic_T_ratio,
        )

        assert abs(float(isentropic_p_ratio(1.0)) - 1.893) < 0.005
        assert abs(float(isentropic_T_ratio(1.0)) - 1.2) < 0.001
        assert abs(float(isentropic_rho_ratio(1.0)) - 1.577) < 0.005

    def test_mach_array(self) -> None:
        from naviertwin.core.flow_analysis.thermofluids.compressible import (
            mach_number,
            speed_of_sound,
        )

        a = speed_of_sound(T=np.array([300, 320]))
        M = mach_number(np.array([100, 100]), a)
        assert M.shape == (2,)
        assert M[0] > M[1]  # 같은 u 일 때 T 높으면 a 커져서 M 작아짐
