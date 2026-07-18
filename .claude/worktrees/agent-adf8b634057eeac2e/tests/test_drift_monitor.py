"""Round 542 — drift monitor."""

from __future__ import annotations

import numpy as np


class TestDrift:
    def test_ks_zero_same(self) -> None:
        from naviertwin.core.twin.drift_monitor import ks_stat

        a = np.linspace(0, 1, 100)
        assert ks_stat(a, a) == 0

    def test_ks_detects_shift(self) -> None:
        from naviertwin.core.twin.drift_monitor import ks_stat

        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 500)
        b = rng.normal(2, 1, 500)
        assert ks_stat(a, b) > 0.4

    def test_psi(self) -> None:
        from naviertwin.core.twin.drift_monitor import psi

        rng = np.random.default_rng(0)
        e = rng.normal(0, 1, 1000)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(2, 1, 1000)
        psi_no_drift = psi(e, a)
        psi_drift = psi(e, b)
        assert psi_drift > psi_no_drift
