"""Round 583 — KrigingSurrogate fallback paths (SMT and sklearn-GP missing)."""

from __future__ import annotations

import builtins

import numpy as np
import pytest


class TestKrigingFallback:
    def test_invalid_X_dim(self) -> None:
        from naviertwin.core.surrogate.kriging_surrogate import KrigingSurrogate

        with pytest.raises(ValueError):
            KrigingSurrogate().fit(np.arange(5), np.arange(5))

    def test_numpy_lstsq_fallback(self, monkeypatch) -> None:
        """Force the deepest fallback (no smt + no sklearn) → numpy lstsq."""
        from naviertwin.core.surrogate.kriging_surrogate import KrigingSurrogate

        real_import = builtins.__import__

        def block_smt_sklearn(name, *a, **kw):
            if name.startswith("smt") or name.startswith("sklearn"):
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block_smt_sklearn)

        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (20, 2))
        y = X[:, 0] + 0.5 * X[:, 1] + 0.01 * rng.standard_normal(20)

        krig = KrigingSurrogate()
        krig.fit(X, y)
        assert krig._backend == "numpy"
        y_pred = krig.predict(X)
        # linear data → near-perfect lstsq fit
        assert np.linalg.norm(y_pred - y) / np.linalg.norm(y) < 0.1
