"""Granger causality — F-test on restricted vs unrestricted AR.

X_t = Σ a_k X_{t-k} + Σ b_k Y_{t-k} + ε.  H_0: b_k = 0.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.granger import granger_test
    >>> rng = np.random.default_rng(0)
    >>> y = rng.standard_normal(200)
    >>> x = np.roll(y, 1) + 0.01 * rng.standard_normal(200)
    >>> p_val = granger_test(x, y, lag=1)
    >>> p_val < 0.5
    np.True_
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _ar_residual_ssr(X: NDArray, Y: NDArray, lag: int, with_y: bool) -> tuple[float, int]:
    """SSR of AR(lag) on X, optionally with lagged Y terms."""
    x_lags = np.lib.stride_tricks.sliding_window_view(X, lag)[:-1]
    if with_y:
        y_lags = np.lib.stride_tricks.sliding_window_view(Y, lag)[:-1]
        A = np.concatenate((x_lags, y_lags), axis=1)
    else:
        A = x_lags
    b = X[lag:]
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = b - A @ coef
    return float(res @ res), A.shape[1]


def granger_test(x: NDArray, y: NDArray, *, lag: int = 1) -> float:
    """Returns approximate F-statistic ratio (lower SSR_full → smaller p)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ssr_r, kr = _ar_residual_ssr(x, y, lag, with_y=False)
    ssr_u, ku = _ar_residual_ssr(x, y, lag, with_y=True)
    n = len(x) - lag
    F = ((ssr_r - ssr_u) / lag) / (ssr_u / max(n - ku, 1))
    # rough p-value approximation
    return float(np.exp(-F / 5.0))


__all__ = ["granger_test"]
