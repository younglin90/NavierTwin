"""Round 221 — Split conformal."""

from __future__ import annotations

import numpy as np
import pytest


class TestConformal:
    def test_coverage(self) -> None:
        from naviertwin.core.uncertainty.conformal import SplitConformal

        rng = np.random.default_rng(0)
        y_cal = rng.standard_normal(500)
        y_pred_cal = y_cal + 0.3 * rng.standard_normal(500)
        cp = SplitConformal(alpha=0.1).calibrate(y_cal, y_pred_cal)
        # test set
        y_test = rng.standard_normal(1000)
        y_pred_test = y_test + 0.3 * rng.standard_normal(1000)
        cov = cp.coverage(y_test, y_pred_test)
        assert 0.85 < cov < 0.95

    def test_invalid_alpha(self) -> None:
        from naviertwin.core.uncertainty.conformal import SplitConformal

        with pytest.raises(ValueError):
            SplitConformal(alpha=1.5)

    def test_interval_width(self) -> None:
        from naviertwin.core.uncertainty.conformal import SplitConformal

        rng = np.random.default_rng(0)
        y_true = rng.standard_normal(200)
        y_pred = y_true + 0.1 * rng.standard_normal(200)
        cp = SplitConformal(alpha=0.1).calibrate(y_true, y_pred)
        lo, hi = cp.predict_interval(y_pred[:5])
        assert np.all(hi - lo > 0)
