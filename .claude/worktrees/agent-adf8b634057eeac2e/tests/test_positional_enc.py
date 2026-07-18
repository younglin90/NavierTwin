"""Round 222 — Fourier / Gaussian RFF encoding."""

from __future__ import annotations

import numpy as np


class TestPE:
    def test_fourier_shape(self) -> None:
        from naviertwin.core.neural.positional_enc import fourier_encode

        x = np.random.default_rng(0).standard_normal((50, 3))
        y = fourier_encode(x, num_freqs=4, include_input=True)
        assert y.shape == (50, 3 + 2 * 4 * 3)

    def test_fourier_no_input(self) -> None:
        from naviertwin.core.neural.positional_enc import fourier_encode

        x = np.zeros((10, 2))
        y = fourier_encode(x, num_freqs=3, include_input=False)
        assert y.shape == (10, 2 * 3 * 2)

    def test_rff_shape(self) -> None:
        from naviertwin.core.neural.positional_enc import gaussian_rff

        x = np.random.default_rng(0).standard_normal((20, 4))
        y = gaussian_rff(x, num_features=64, sigma=1.0, seed=0)
        assert y.shape == (20, 64)

    def test_rff_bounded(self) -> None:
        from naviertwin.core.neural.positional_enc import gaussian_rff

        x = np.random.default_rng(0).standard_normal((10, 2))
        y = gaussian_rff(x, num_features=32)
        # cos 값 * sqrt(2/F), bounded by sqrt(2/F)
        assert np.all(np.abs(y) <= np.sqrt(2 / 32) + 1e-12)
