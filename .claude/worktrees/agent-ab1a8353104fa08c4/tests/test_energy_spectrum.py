"""Round 185 — 에너지 스펙트럼."""

from __future__ import annotations

import numpy as np


class TestSpectrum:
    def test_1d_peak_at_mode(self) -> None:
        from naviertwin.core.analysis.energy_spectrum import energy_spectrum_1d

        n = 256
        L = 1.0
        t = np.linspace(0, L, n, endpoint=False)
        mode = 7
        u = np.sin(2 * np.pi * mode * t)
        k, E = energy_spectrum_1d(u, L)
        peak_k = k[np.argmax(E)]
        assert abs(peak_k - mode) < 0.5

    def test_2d_radial(self) -> None:
        from naviertwin.core.analysis.energy_spectrum import (
            energy_spectrum_2d_radial,
        )

        n = 64
        rng = np.random.default_rng(0)
        u = rng.standard_normal((n, n))
        v = rng.standard_normal((n, n))
        k, E = energy_spectrum_2d_radial(u, v, Lx=1, Ly=1, n_bins=16)
        assert k.shape == E.shape
        assert E.sum() > 0

    def test_shape_mismatch(self) -> None:
        import pytest as pt

        from naviertwin.core.analysis.energy_spectrum import (
            energy_spectrum_2d_radial,
        )

        with pt.raises(ValueError):
            energy_spectrum_2d_radial(np.zeros((3, 3)), np.zeros((3, 4)))
