"""Round 124 — 필드 정합성."""

from __future__ import annotations

import numpy as np


class TestFieldSanity:
    def test_sanity(self) -> None:
        from naviertwin.core.validation.field_sanity import field_sanity_check

        x = np.array([1.0, 2.0, np.nan, 4.0, np.inf, -np.inf])
        r = field_sanity_check(x)
        assert r["n_nan"] == 1
        assert r["n_inf"] == 2
        assert r["all_finite"] is False
        assert r["n_finite"] == 3

    def test_range(self) -> None:
        from naviertwin.core.validation.field_sanity import field_sanity_check

        x = np.array([0.0, 5.0, 10.0, 15.0])
        r = field_sanity_check(x, expected_range=(0, 10))
        assert r["n_above"] == 1
        assert r["in_range"] is False

    def test_iqr_outliers(self) -> None:
        from naviertwin.core.validation.field_sanity import detect_outliers_iqr

        rng = np.random.default_rng(0)
        x = np.concatenate([rng.standard_normal(100), np.array([50.0, -50.0])])
        mask = detect_outliers_iqr(x)
        assert mask[-1] and mask[-2]
        assert mask.sum() >= 2

    def test_zscore_outliers(self) -> None:
        from naviertwin.core.validation.field_sanity import detect_outliers_zscore

        rng = np.random.default_rng(0)
        x = np.concatenate([rng.standard_normal(200), np.array([20.0])])
        mask = detect_outliers_zscore(x, threshold=3.0)
        assert mask[-1]
