"""Round 182 — delay embedding."""

from __future__ import annotations

import numpy as np


class TestDelayEmbed:
    def test_shape(self) -> None:
        from naviertwin.core.system_id.delay_embed import delay_embed

        x = np.linspace(0, 1, 100)
        Y = delay_embed(x, dim=4, delay=5)
        assert Y.shape == (100 - 3 * 5, 4)

    def test_invalid(self) -> None:
        import pytest as pt

        from naviertwin.core.system_id.delay_embed import delay_embed

        with pt.raises(ValueError):
            delay_embed(np.zeros(5), dim=4, delay=5)

    def test_autocorrelation(self) -> None:
        from naviertwin.core.system_id.delay_embed import (
            autocorrelation,
            first_zero_crossing,
        )

        t = np.linspace(0, 20, 2000)
        x = np.sin(t)
        ac = autocorrelation(x, max_lag=400)
        assert abs(ac[0] - 1.0) < 1e-9
        # sin 주기 2π → zero crossing ≈ T/4
        zc = first_zero_crossing(ac)
        assert zc > 5
