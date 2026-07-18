"""Round 110 — probe 시계열."""

from __future__ import annotations

import numpy as np
import pytest


class TestProbe:
    def test_nearest(self) -> None:
        from naviertwin.core.analysis.probe import probe_time_series

        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        X = np.array([
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ], dtype=float)
        probes = np.array([[1.1, 0, 0], [1.9, 0, 0]])
        ts = probe_time_series(X, coords, probes, method="nearest")
        assert ts[0].tolist() == [20, 21, 22, 23]
        assert ts[1].tolist() == [30, 31, 32, 33]

    def test_idw(self) -> None:
        from naviertwin.core.analysis.probe import probe_time_series

        coords = np.array([[0, 0, 0], [2, 0, 0]], dtype=float)
        X = np.array([[0, 0], [10, 10]], dtype=float)
        probes = np.array([[1.0, 0, 0]])
        ts = probe_time_series(X, coords, probes, method="idw", k=2)
        assert abs(ts[0, 0] - 5.0) < 1e-10

    def test_stats(self) -> None:
        from naviertwin.core.analysis.probe import probe_statistics

        ts = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        s = probe_statistics(ts)
        assert s["mean"].tolist() == [2.0, 20.0]
        assert s["min"].tolist() == [1.0, 10.0]
        assert s["max"].tolist() == [3.0, 30.0]

    def test_invalid_method(self) -> None:
        from naviertwin.core.analysis.probe import probe_time_series

        with pytest.raises(ValueError):
            probe_time_series(
                np.zeros((3, 2)), np.zeros((3, 3)), np.zeros((1, 3)),
                method="bogus",
            )
