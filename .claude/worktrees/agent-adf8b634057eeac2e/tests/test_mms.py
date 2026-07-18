"""Round 501 — MMS."""

from __future__ import annotations

import numpy as np


class TestMMS:
    def test_poisson_source(self) -> None:
        from naviertwin.core.verification.mms import poisson_1d_source

        # u = sin(πx), -u'' = π² sin(πx)
        x = np.linspace(0, 1, 21)
        S = poisson_1d_source(x, lambda x: np.sin(np.pi * x), eps=1e-3)
        expected = np.pi ** 2 * np.sin(np.pi * x)
        assert np.max(np.abs(S - expected)) < 0.05

    def test_l2_error(self) -> None:
        from naviertwin.core.verification.mms import l2_error

        u = np.array([1.0, 2.0, 3.0])
        v = np.array([1.1, 2.0, 3.0])
        err = l2_error(u, v, dx=1.0)
        assert abs(err - 0.1) < 1e-9
