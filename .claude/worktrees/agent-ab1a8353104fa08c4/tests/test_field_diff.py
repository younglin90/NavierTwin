"""Round 82 — field diff stats."""

from __future__ import annotations

import numpy as np
import pytest


class TestFieldDiff:
    def test_stats(self) -> None:
        from naviertwin.core.validation.field_diff import field_diff_stats

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 1.9, 3.0])
        s = field_diff_stats(a, b)
        assert s["mae"] == pytest.approx(np.mean([0.1, 0.1, 0.0]))
        assert s["max_abs"] == pytest.approx(0.1)
        assert s["relative_l2"] > 0

    def test_shape_mismatch(self) -> None:
        from naviertwin.core.validation.field_diff import field_diff_stats

        with pytest.raises(ValueError):
            field_diff_stats(np.zeros(3), np.zeros(4))

    def test_hotspot_and_mask(self) -> None:
        from naviertwin.core.validation.field_diff import band_mask, hotspot_indices

        a = np.zeros(10)
        b = np.array([0, 0, 5, 0, 0, 3, 0, 0, 2, 0], dtype=float)
        idx = hotspot_indices(a, b, top_k=3)
        assert set(idx.tolist()) == {2, 5, 8}
        assert idx[0] == 2

        m = band_mask(a, b, tol=1.0)
        assert m.sum() == 3
