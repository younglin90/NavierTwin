"""Round 128 — spectral 미분 / 필터."""

from __future__ import annotations

import numpy as np


class TestSpectral:
    def test_sin_derivative(self) -> None:
        from naviertwin.core.analysis.spectral import spectral_derivative_1d

        n = 128
        L = 2 * np.pi
        x = np.linspace(0, L, n, endpoint=False)
        f = np.sin(3 * x)
        df = spectral_derivative_1d(f, L, order=1)
        assert np.max(np.abs(df - 3 * np.cos(3 * x))) < 1e-8

    def test_second_derivative(self) -> None:
        from naviertwin.core.analysis.spectral import spectral_derivative_1d

        n = 128
        L = 2 * np.pi
        x = np.linspace(0, L, n, endpoint=False)
        f = np.sin(2 * x)
        d2f = spectral_derivative_1d(f, L, order=2)
        assert np.max(np.abs(d2f - (-4 * np.sin(2 * x)))) < 1e-8

    def test_derivative_2d(self) -> None:
        from naviertwin.core.analysis.spectral import spectral_derivative_2d

        n = 64
        L = 2 * np.pi
        x = np.linspace(0, L, n, endpoint=False)
        y = np.linspace(0, L, n, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="xy")
        f = np.sin(X) * np.cos(Y)
        dfdx = spectral_derivative_2d(f, L, L, order=(1, 0))
        assert np.max(np.abs(dfdx - np.cos(X) * np.cos(Y))) < 1e-8

    def test_lowpass(self) -> None:
        from naviertwin.core.analysis.spectral import lowpass_filter_1d

        n = 256
        L = 2 * np.pi
        x = np.linspace(0, L, n, endpoint=False)
        f = np.sin(2 * x) + np.sin(20 * x)
        y = lowpass_filter_1d(f, L, cutoff=5.0)
        # 고주파 제거 → sin(2x) 만 남음
        assert np.max(np.abs(y - np.sin(2 * x))) < 1e-8
