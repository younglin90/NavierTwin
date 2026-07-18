"""Round 234 — ISOMAP."""

from __future__ import annotations

import numpy as np


class TestISO:
    def test_swiss_roll_1d(self) -> None:
        """Swiss roll 1D unroll → 첫 성분이 매개변수 t 와 강한 상관."""
        from naviertwin.core.dimensionality_reduction.nonlinear.isomap import isomap

        t = np.linspace(0, 4 * np.pi, 150)
        X = np.stack([t * np.cos(t), t * np.sin(t)], axis=1)
        Y = isomap(X, k=8, n_components=1)
        corr = float(abs(np.corrcoef(Y[:, 0], t)[0, 1]))
        assert corr > 0.9

    def test_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.isomap import isomap

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 4))
        Y = isomap(X, k=5, n_components=2)
        assert Y.shape == (50, 2)
