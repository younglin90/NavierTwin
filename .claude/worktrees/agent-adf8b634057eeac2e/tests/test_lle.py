"""Round 235 — LLE."""

from __future__ import annotations

import numpy as np


class TestLLE:
    def test_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.lle import lle

        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 4))
        Y = lle(X, k=8, n_components=2)
        assert Y.shape == (80, 2)

    def test_curve(self) -> None:
        """원형 curve 를 1D 로 embed."""
        from naviertwin.core.dimensionality_reduction.nonlinear.lle import lle

        t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        X = np.stack([np.cos(t), np.sin(t)], axis=1)
        Y = lle(X, k=8, n_components=1)
        # periodic → 1D embed 엄밀 증명 힘들지만 finite 확인
        assert np.all(np.isfinite(Y))
