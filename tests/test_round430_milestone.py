"""Round 430 — Q category milestone: time-series / digital twin (R421-R429) e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneQ:
    def test_imports(self) -> None:
        from naviertwin.core.analysis import (  # noqa: F401
            anomaly_ae,
            cusum,
            granger,
            hilbert_spectrum,
            synchrosqueeze,
        )
        from naviertwin.core.data_assimilation import (  # noqa: F401
            ks_nonlinear,
            particle_smoother,
        )
        from naviertwin.core.system_id import lstm_forecast, var_id  # noqa: F401

    def test_anomaly_cusum_pipeline(self) -> None:
        from naviertwin.core.analysis.cusum import cusum_detect

        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 100)])
        idx = cusum_detect(x, threshold=5.0, mean=0.0, sigma=1.0, k=0.5)
        assert 0 <= idx < 200  # detection occurs within the series
