"""RBF-FD — derivative weights from local stencil + Gaussian RBF.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.meshless.rbf_fd import rbf_fd_weights_1d
    >>> stencil = np.array([-1.0, 0.0, 1.0])
    >>> w = rbf_fd_weights_1d(stencil, eps=1.0, order=1)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _rbf(r2, eps):
    return np.exp(-eps * r2)


def rbf_fd_weights_1d(
    stencil: NDArray[np.float64], *, eps: float = 1.0, order: int = 1,
    center: float = 0.0,
) -> NDArray[np.float64]:
    """Weights w such that Σ w_i f(x_i) ≈ f^(order)(center)."""
    s = np.asarray(stencil, dtype=np.float64)
    n = len(s)
    A = _rbf((s[:, None] - s[None, :]) ** 2, eps)
    # RHS: derivative of RBF at center
    if order == 1:
        rhs = -2 * eps * (center - s) * _rbf((center - s) ** 2, eps)
    elif order == 2:
        diff = center - s
        rhs = (-2 * eps + 4 * eps * eps * diff ** 2) * _rbf(diff ** 2, eps)
    else:
        raise NotImplementedError("order ≤ 2")
    w = np.linalg.solve(A, rhs)
    _ = n
    return w


__all__ = ["rbf_fd_weights_1d"]
