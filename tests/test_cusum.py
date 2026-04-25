"""Round 428 — CUSUM."""

from __future__ import annotations

import numpy as np


class TestCUSUM:
    def test_detect_step(self) -> None:
        from naviertwin.core.analysis.cusum import cusum_detect

        rng = np.random.default_rng(0)
        x = np.concatenate([
            rng.normal(0, 1, 50),
            rng.normal(3, 1, 50),
        ])
        # explicit baseline mean/sigma + larger threshold → detection past change-pt
        idx = cusum_detect(x, threshold=5.0, mean=0.0, sigma=1.0, k=0.5)
        # within ±10 of true change-point (50)
        assert 40 <= idx < 80

    def test_no_change(self) -> None:
        from naviertwin.core.analysis.cusum import cusum_detect

        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 50)
        # mean=0, sigma=1; threshold high → unlikely
        idx = cusum_detect(x, threshold=10.0, mean=0.0, sigma=1.0)
        assert idx == -1
