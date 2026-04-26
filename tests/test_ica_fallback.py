"""Round 582 — coverage uplift for FastICA numpy fallback path."""

from __future__ import annotations

import builtins

import numpy as np
import pytest


class TestFastICAFallback:
    def test_2d_input_required(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.ica import FastICA

        with pytest.raises(ValueError, match="2D"):
            FastICA(n_components=2).fit_transform(np.zeros(10))

    def test_numpy_fallback_runs(self, monkeypatch) -> None:
        """Force the numpy fallback by blocking sklearn.decomposition import."""
        from naviertwin.core.dimensionality_reduction.linear.ica import FastICA

        real_import = builtins.__import__

        def block_sklearn(name, *a, **kw):
            if name.startswith("sklearn"):
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block_sklearn)

        rng = np.random.default_rng(0)
        t = np.linspace(0, 8, 200)
        s1 = np.sin(2 * t)
        s2 = np.sign(np.sin(3 * t))
        S = np.column_stack([s1, s2])
        A = np.array([[1.0, 0.6], [0.4, 1.0]])
        X = S @ A.T + 0.01 * rng.standard_normal(S.shape)

        ica = FastICA(n_components=2, max_iter=50, seed=0)
        S_rec = ica.fit_transform(X)
        assert S_rec.shape == (200, 2)
        assert ica.is_fitted
        assert ica.W_ is not None
        assert ica.whitening_ is not None
        assert ica.mean_ is not None
