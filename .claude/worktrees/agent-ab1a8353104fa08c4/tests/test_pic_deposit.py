"""Round 473 — PIC CIC deposition."""

from __future__ import annotations

import numpy as np


class TestPIC:
    def test_at_node(self) -> None:
        from naviertwin.core.meshless.pic_deposit import deposit_cic_1d

        rho = deposit_cic_1d(np.array([2.0]), np.array([1.0]), n_grid=5, dx=1.0)
        # particle at grid node 2 → all to that node
        assert np.allclose(rho, [0, 0, 1.0, 0, 0])

    def test_between(self) -> None:
        from naviertwin.core.meshless.pic_deposit import deposit_cic_1d

        rho = deposit_cic_1d(np.array([1.5]), np.array([1.0]), n_grid=4, dx=1.0)
        # split 50/50
        assert np.allclose(rho, [0, 0.5, 0.5, 0])

    def test_total_charge(self) -> None:
        from naviertwin.core.meshless.pic_deposit import deposit_cic_1d

        rho = deposit_cic_1d(
            np.array([0.3, 1.7, 2.4]), np.array([1.0, 1.0, 1.0]),
            n_grid=5, dx=1.0,
        )
        assert abs(rho.sum() - 3.0) < 1e-12
