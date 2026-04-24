"""Round 55 — SMT KPLS/IDW/QP/DOE 래퍼."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("smt", reason="smt 필요")


class TestDOE:
    def test_lhs(self) -> None:
        from naviertwin.core.surrogate.smt_advanced import lhs_design

        xlim = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        X = lhs_design(n=20, d=2, xlimits=xlim, seed=0)
        assert X.shape == (20, 2)
        assert np.all(X >= -1) and np.all(X <= 1)

    def test_full_factorial(self) -> None:
        from naviertwin.core.surrogate.smt_advanced import full_factorial

        xlim = np.array([[0.0, 1.0], [0.0, 1.0]])
        X = full_factorial(n_per_axis=4, xlimits=xlim)
        assert X.shape[1] == 2


class TestKPLS:
    def test_fit_predict(self) -> None:
        from naviertwin.core.surrogate.smt_advanced import kpls_fit, smt_predict

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 5))
        y = (X[:, 0] ** 2 + 0.5 * X[:, 1]).reshape(-1, 1)
        model = kpls_fit(X, y, n_comp=2)
        y_pred = smt_predict(model, X)
        err = float(np.linalg.norm(y - y_pred) / np.linalg.norm(y))
        assert err < 0.5


class TestIDW:
    def test_fit_predict(self) -> None:
        from naviertwin.core.surrogate.smt_advanced import idw_fit, smt_predict

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 2))
        y = (X[:, 0] + X[:, 1]).reshape(-1, 1)
        model = idw_fit(X, y, p=2.0)
        y_pred = smt_predict(model, X)
        # IDW: 학습점에서 정확
        assert np.allclose(y_pred, y, atol=1e-6)


class TestQP:
    def test_fit_predict(self) -> None:
        from naviertwin.core.surrogate.smt_advanced import qp_fit, smt_predict

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 2))
        y = (X[:, 0] ** 2 + X[:, 1] ** 2).reshape(-1, 1)
        model = qp_fit(X, y)
        y_pred = smt_predict(model, X)
        # 2차 다항식 → 2차 함수 정확 피팅
        assert np.allclose(y_pred, y, atol=1e-6)
