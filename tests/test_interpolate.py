"""Round 109 — KNN / IDW 보간."""

from __future__ import annotations

import numpy as np


class TestInterpolate:
    def test_knn_exact(self) -> None:
        from naviertwin.core.analysis.interpolate import knn_interpolate

        src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        vals = np.array([10.0, 20.0, 30.0])
        tgt = np.array([[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]])
        y = knn_interpolate(src, vals, tgt, k=1)
        assert y[0] == 10.0
        assert y[1] == 30.0

    def test_idw_midpoint(self) -> None:
        from naviertwin.core.analysis.interpolate import idw_interpolate

        src = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        vals = np.array([0.0, 10.0])
        y = idw_interpolate(src, vals, np.array([[1.0, 0.0, 0.0]]))
        assert abs(y[0] - 5.0) < 1e-10

    def test_idw_k(self) -> None:
        from naviertwin.core.analysis.interpolate import idw_interpolate

        rng = np.random.default_rng(0)
        src = rng.standard_normal((50, 3))
        vals = rng.standard_normal(50)
        tgt = rng.standard_normal((5, 3))
        y = idw_interpolate(src, vals, tgt, k=5)
        assert y.shape == (5,)
        assert np.all(np.isfinite(y))

    def test_knn_shape(self) -> None:
        from naviertwin.core.analysis.interpolate import knn_interpolate

        rng = np.random.default_rng(0)
        src = rng.standard_normal((20, 3))
        vals = rng.standard_normal(20)
        tgt = rng.standard_normal((4, 3))
        y = knn_interpolate(src, vals, tgt, k=3)
        assert y.shape == (4,)
