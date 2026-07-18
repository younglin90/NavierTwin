"""Round 604 — MRPOD multi-resolution POD coverage."""

from __future__ import annotations

import numpy as np
import pytest


class TestMRPOD:
    def _make_data(self, n_space: int = 80, n_snap: int = 40, seed: int = 0):
        rng = np.random.default_rng(seed)
        # multi-scale signal
        x = np.linspace(0, 4 * np.pi, n_space)
        t = np.linspace(0, 2 * np.pi, n_snap)
        # coarse scale + fine scale
        X = np.outer(np.sin(x), np.sin(t)) + 0.2 * np.outer(np.sin(5 * x), np.cos(3 * t))
        X += 0.01 * rng.standard_normal((n_space, n_snap))
        return X

    def test_fit_basic(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD

        X = self._make_data()
        m = MRPOD(n_scales=3, n_modes_per_scale=4)
        m.fit(X)
        assert m.is_fitted
        assert len(m.scale_modes) == 3
        assert len(m.scale_energies) == 3

    def test_get_modes_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD

        X = self._make_data()
        m = MRPOD(n_scales=2, n_modes_per_scale=3)
        m.fit(X)
        modes = m.get_modes()
        assert modes.shape[0] == 80
        assert modes.shape[1] <= 6  # 2 * 3

    def test_energy_fraction_sums_to_one(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD

        X = self._make_data()
        m = MRPOD(n_scales=3, n_modes_per_scale=4)
        m.fit(X)
        ef = m.get_energy_fraction()
        assert ef.shape == (3,)
        assert abs(ef.sum() - 1.0) < 1e-10

    def test_reconstruct(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD

        X = self._make_data()
        m = MRPOD(n_scales=2, n_modes_per_scale=3)
        m.fit(X)
        modes = m.get_modes()
        n_modes = modes.shape[1]
        coeffs = np.ones(n_modes)
        rec = m.reconstruct(coeffs)
        assert rec.shape == (80,)

    def test_reconstruct_wrong_coeff_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD

        X = self._make_data()
        m = MRPOD(n_scales=2, n_modes_per_scale=3)
        m.fit(X)
        with pytest.raises(ValueError, match="coefficients shape"):
            m.reconstruct(np.zeros(999))

    def test_fit_1d_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD

        m = MRPOD(n_scales=2)
        with pytest.raises(ValueError, match="2D"):
            m.fit(np.zeros(80))

    def test_invalid_n_scales_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD

        with pytest.raises(ValueError, match="n_scales"):
            MRPOD(n_scales=0)

    def test_invalid_n_modes_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD

        with pytest.raises(ValueError, match="n_modes_per_scale"):
            MRPOD(n_modes_per_scale=0)

    def test_get_modes_before_fit_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD

        m = MRPOD()
        with pytest.raises(RuntimeError, match="fit"):
            m.get_modes()

    def test_get_energy_fraction_before_fit_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD

        m = MRPOD()
        with pytest.raises(RuntimeError, match="fit"):
            m.get_energy_fraction()

    def test_zero_energy_fraction(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD

        X = np.zeros((20, 10))
        m = MRPOD(n_scales=2, n_modes_per_scale=2)
        m.fit(X)
        ef = m.get_energy_fraction()
        assert np.all(ef == 0.0)
