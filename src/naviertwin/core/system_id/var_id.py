"""VAR(p) identification — least-squares vector autoregression.

X_t = Σ_{k=1}^p A_k X_{t-k} + ε.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.system_id.var_id import fit_var
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 2))
    >>> As = fit_var(X, p=2)
    >>> len(As)
    2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fit_var(X: NDArray[np.float64], p: int = 1) -> list[NDArray[np.float64]]:
    """Stack lags → solve A_full = LSQ([X_{t-1} ... X_{t-p}], X_t)."""
    X = np.asarray(X, dtype=np.float64)
    T, d = X.shape
    Y = X[p:]                 # (T-p, d)
    Z_blocks = []
    k = 0
    while k < p:
        Z_blocks.append(X[p - k - 1:T - k - 1])
        k += 1
    Z = np.hstack(Z_blocks)   # (T-p, p*d)
    A_full, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    A_full = A_full.T  # (d, p*d)
    blocks = []
    k = 0
    while k < p:
        blocks.append(A_full[:, k * d:(k + 1) * d])
        k += 1
    return blocks


__all__ = ["fit_var"]
