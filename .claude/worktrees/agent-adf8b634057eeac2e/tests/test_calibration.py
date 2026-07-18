"""Round 497 — calibration."""

from __future__ import annotations

import numpy as np


class TestCalibration:
    def test_ece_zero_perfect(self) -> None:
        from naviertwin.utils.calibration import ece

        # all predictions confident and correct → ECE ≈ 0
        n = 100
        probs = np.zeros((n, 2))
        probs[:, 0] = 1.0
        labels = np.zeros(n, dtype=int)
        assert ece(probs, labels) == 0.0

    def test_temperature_scaling(self) -> None:
        from naviertwin.utils.calibration import temperature_scale

        rng = np.random.default_rng(0)
        n = 200
        # over-confident logits
        logits = 5.0 * rng.standard_normal((n, 3))
        labels = np.argmax(logits, axis=1)
        # introduce errors
        labels = np.where(rng.random(n) < 0.2, (labels + 1) % 3, labels)
        T, p_cal = temperature_scale(logits, labels)
        assert T > 0
        assert p_cal.shape == (n, 3)
