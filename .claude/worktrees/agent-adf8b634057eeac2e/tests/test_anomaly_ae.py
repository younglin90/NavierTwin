"""Round 427 — anomaly AE."""

from __future__ import annotations

import numpy as np


class TestAnomaly:
    def test_detects_outlier(self) -> None:
        from naviertwin.core.analysis.anomaly_ae import POD_AnomalyDetector

        rng = np.random.default_rng(0)
        # low-rank training data
        Z = rng.standard_normal((50, 3))
        E = rng.standard_normal((10, 3))
        X = E @ Z.T  # (10, 50)
        det = POD_AnomalyDetector(rank=3).fit(X)
        # in-distribution
        assert det.score(X[:, 0]).item() < 1e-9
        # outlier (random direction)
        outlier = rng.standard_normal(10) * 10
        assert det.is_anomaly(outlier)
