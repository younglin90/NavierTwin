"""Round 179 — anomaly detection."""

from __future__ import annotations

import numpy as np


class TestAnomaly:
    def test_threshold(self) -> None:
        from naviertwin.core.monitoring.anomaly import threshold_detector

        x = np.array([0.1, 0.2, 3.0, 0.0, -4.0])
        m = threshold_detector(x, threshold=2.0)
        assert m.tolist() == [False, False, True, False, True]

    def test_zscore(self) -> None:
        from naviertwin.core.monitoring.anomaly import zscore_detector

        rng = np.random.default_rng(0)
        x = np.concatenate([rng.standard_normal(200), np.array([10.0])])
        m = zscore_detector(x, k=3.0)
        assert m[-1]

    def test_cusum_detects_drift(self) -> None:
        from naviertwin.core.monitoring.anomaly import cusum

        # step shift after index 100
        x = np.concatenate([np.zeros(100), np.ones(50) * 1.0])
        alarms = cusum(x, k=0.3, h=2.0)
        # 후반에 알람
        assert alarms[100:].any()
        assert not alarms[:80].any()

    def test_ewma(self) -> None:
        from naviertwin.core.monitoring.anomaly import ewma

        rng = np.random.default_rng(0)
        x = np.concatenate([rng.standard_normal(100) * 0.1, np.ones(20) * 2.0])
        alarms = ewma(x, lam=0.3, k=2.0)
        assert alarms[100:].any()
