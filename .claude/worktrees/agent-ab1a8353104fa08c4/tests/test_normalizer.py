"""Round 87 — Normalizer (minmax/standard/robust/maxabs)."""

from __future__ import annotations

import numpy as np
import pytest


class TestNormalizer:
    @pytest.mark.parametrize("kind", ["minmax", "standard", "robust", "maxabs"])
    def test_roundtrip(self, kind: str) -> None:
        from naviertwin.core.preprocessing.normalizer import Normalizer

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 4)) * 10.0 + 5.0
        n = Normalizer(kind)
        Y = n.fit_transform(X)
        Xr = n.inverse_transform(Y)
        assert np.allclose(X, Xr, atol=1e-9)

    def test_standard_zero_mean(self) -> None:
        from naviertwin.core.preprocessing.normalizer import Normalizer

        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3)) * 3.0 + 5.0
        Y = Normalizer("standard").fit_transform(X)
        assert np.allclose(Y.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(Y.std(axis=0), 1.0, atol=1e-10)

    def test_minmax_range(self) -> None:
        from naviertwin.core.preprocessing.normalizer import Normalizer

        rng = np.random.default_rng(0)
        X = rng.uniform(-3, 7, size=(80, 2))
        Y = Normalizer("minmax").fit_transform(X)
        assert np.allclose(Y.min(axis=0), 0.0)
        assert np.allclose(Y.max(axis=0), 1.0)

    def test_invalid_kind(self) -> None:
        from naviertwin.core.preprocessing.normalizer import Normalizer

        with pytest.raises(ValueError):
            Normalizer("bogus")

    def test_not_fitted(self) -> None:
        from naviertwin.core.preprocessing.normalizer import Normalizer

        with pytest.raises(RuntimeError):
            Normalizer("standard").transform(np.zeros((3, 2)))

    def test_constant_column(self) -> None:
        from naviertwin.core.preprocessing.normalizer import Normalizer

        X = np.ones((10, 2)) * 5.0
        Y = Normalizer("standard").fit_transform(X)  # scale=1 fallback
        assert np.all(np.isfinite(Y))
