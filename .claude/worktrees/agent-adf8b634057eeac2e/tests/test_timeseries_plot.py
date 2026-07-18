"""Round 434 — time-series plot helper."""

from __future__ import annotations

import numpy as np


class TestTSPlot:
    def test_band(self) -> None:
        from naviertwin.core.visualization.timeseries_plot import series_with_band

        t = np.linspace(0, 1, 5)
        y = np.zeros(5)
        s = np.ones(5)
        d = series_with_band(t, y, s, z=1.0)
        assert np.allclose(d["upper"], 1.0)
        assert np.allclose(d["lower"], -1.0)

    def test_downsample(self) -> None:
        from naviertwin.core.visualization.timeseries_plot import downsample

        t = np.arange(10000)
        y = np.zeros(10000)
        t2, y2 = downsample(t, y, max_points=500)
        assert len(t2) <= 500 + 50
