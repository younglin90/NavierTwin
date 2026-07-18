"""Round 500 — X category milestone: learning theory (R491-R499) e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneX:
    def test_imports(self) -> None:
        from naviertwin.utils import (  # noqa: F401
            calibration,
            grad_confusion,
            influence,
            lipschitz,
            mi_knn,
            ntk,
            pac_bayes,
            sam,
            spectral_norm,
        )

    def test_calibration_e2e(self) -> None:
        from naviertwin.utils.calibration import ece, temperature_scale

        rng = np.random.default_rng(0)
        n = 200
        logits = 5.0 * rng.standard_normal((n, 3))
        labels = np.argmax(logits, axis=1)
        labels = np.where(rng.random(n) < 0.3, (labels + 1) % 3, labels)
        T, p_cal = temperature_scale(logits, labels)
        # raw probs vs calibrated probs ECE comparison
        from naviertwin.utils.calibration import _softmax
        p_raw = _softmax(logits)
        e_raw = ece(p_raw, labels)
        e_cal = ece(p_cal, labels)
        assert e_cal <= e_raw + 0.05  # calibration should not make it worse much
