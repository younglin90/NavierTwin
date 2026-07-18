"""Round 361 — CHT coupling."""

from __future__ import annotations

import numpy as np


class TestCHT:
    def test_interface_matched(self) -> None:
        from naviertwin.core.coupling.cht import cht_iterate

        Ts = np.linspace(300, 400, 6)
        Tf = np.linspace(500, 400, 6)
        Ts2, Tf2 = cht_iterate(Ts, Tf, k_s=10.0, k_f=1.0, n_iter=200)
        # interface continuous
        assert abs(Ts2[-1] - Tf2[0]) < 1e-9

    def test_finite(self) -> None:
        from naviertwin.core.coupling.cht import cht_iterate

        Ts = np.zeros(8)
        Ts[0] = 100.0
        Tf = np.zeros(8)
        Tf[-1] = 50.0
        Ts2, Tf2 = cht_iterate(Ts, Tf, k_s=5.0, k_f=2.0, n_iter=100)
        assert np.isfinite(Ts2).all() and np.isfinite(Tf2).all()
