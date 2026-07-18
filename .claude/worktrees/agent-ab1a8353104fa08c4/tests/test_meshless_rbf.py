"""Round 475 — meshless RBF interp."""

from __future__ import annotations

import numpy as np


class TestMeshlessRBF:
    def test_exact_at_centers(self) -> None:
        from naviertwin.core.meshless.rbf_interp import RBFInterp

        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 4.0])
        r = RBFInterp(eps=1.0).fit(X, y)
        y_pred = r.predict(X)
        assert np.allclose(y_pred, y, atol=1e-6)

    def test_smooth(self) -> None:
        from naviertwin.core.meshless.rbf_interp import RBFInterp

        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = np.sin(2 * np.pi * X.ravel())
        r = RBFInterp(eps=20.0).fit(X, y)
        Xq = np.linspace(0.1, 0.9, 50).reshape(-1, 1)
        yq = r.predict(Xq)
        assert np.max(np.abs(yq - np.sin(2 * np.pi * Xq.ravel()))) < 0.3
