"""Round 240 — 마일스톤 R231-R239."""

from __future__ import annotations

import numpy as np
import pytest

R231_239 = [
    "naviertwin.core.analysis.particle_tracker",
    "naviertwin.core.surrogate.kriging_scratch",
    "naviertwin.core.surrogate.multi_fidelity",
    "naviertwin.core.dimensionality_reduction.nonlinear.isomap",
    "naviertwin.core.dimensionality_reduction.nonlinear.lle",
    "naviertwin.core.validation.cross_val",
    "naviertwin.utils.benchmark",
    "naviertwin.utils.structured_log",
    "naviertwin.utils.parallel",
]


class TestRound240:
    @pytest.mark.parametrize("m", R231_239)
    def test_importable(self, m: str) -> None:
        import importlib
        importlib.import_module(m)

    def test_kriging_kfold(self) -> None:
        """Kriging 회귀의 k-fold 오차 검증."""
        from naviertwin.core.surrogate.kriging_scratch import OrdinaryKriging
        from naviertwin.core.validation.cross_val import kfold_scores

        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (40, 1))
        y = np.sin(3 * X[:, 0])

        def fp(X_tr, y_tr, X_val):
            k = OrdinaryKriging(theta=0.3, nugget=1e-6).fit(X_tr, y_tr)
            return k.predict(X_val)

        def mse(yt, yp):
            return float(np.mean((yt - yp) ** 2))

        scores = kfold_scores(X, y, fp, mse, k=4)
        assert len(scores) == 4
        assert np.mean(scores) < 0.2
