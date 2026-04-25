"""Round 495 — MI kNN."""

from __future__ import annotations

import numpy as np


class TestMIKNN:
    def test_self_higher_than_independent(self) -> None:
        from naviertwin.utils.mi_knn import mi_knn_1d

        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        mi_self = mi_knn_1d(x, x + 1e-3 * rng.standard_normal(200))
        mi_indep = mi_knn_1d(x, rng.standard_normal(200))
        assert mi_self > mi_indep
