"""Round 233 — multi-fidelity surrogate."""

from __future__ import annotations

import numpy as np


class TestMF:
    def test_fits_biased_low(self) -> None:
        """low-fi = 1.5 * hi + 0.1 sin. LMF 가 bias 를 ρ 로 교정."""
        from naviertwin.core.surrogate.multi_fidelity import LinearMultiFidelity

        rng = np.random.default_rng(0)
        X_lo = rng.uniform(-1, 1, (40, 1))
        X_hi = rng.uniform(-1, 1, (10, 1))
        y_hi_fn = lambda x: np.sin(3 * x[:, 0])  # noqa: E731
        y_lo_fn = lambda x: 0.8 * np.sin(3 * x[:, 0]) + 0.1  # biased  # noqa: E731
        y_lo = y_lo_fn(X_lo)
        y_hi = y_hi_fn(X_hi)
        mf = LinearMultiFidelity(theta=0.3, nugget=1e-6).fit(X_lo, y_lo, X_hi, y_hi)
        X_test = np.linspace(-0.5, 0.5, 20)[:, None]
        y_pred = mf.predict(X_test)
        y_true = y_hi_fn(X_test)
        # MF 예측이 합리적 수준
        assert np.max(np.abs(y_pred - y_true)) < 0.5
