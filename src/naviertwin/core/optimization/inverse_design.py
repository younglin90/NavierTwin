"""역설계 — gradient 기반 design param 최적화 (regularization 옵션).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.inverse_design import inverse_design
    >>> # target: surrogate output y_target → 설계 p 찾기
    >>> # forward(p) = p**2 + bias, target = 4 → p ≈ 2
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def inverse_design(
    forward: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    target: NDArray[np.float64],
    p0: NDArray[np.float64],
    *,
    reg: float = 0.0,
    lr: float = 0.05,
    n_iter: int = 500,
    bounds: tuple[NDArray | None, NDArray | None] | None = None,
    eps: float = 1e-6,
) -> tuple[NDArray[np.float64], list[float]]:
    """min_p ‖forward(p) - target‖² + reg · ‖p‖²."""
    p = np.asarray(p0, dtype=np.float64).ravel().copy()
    target = np.asarray(target, dtype=np.float64).ravel()
    history = []
    it = 0
    while it < n_iter:
        y = forward(p)
        r = y - target
        J = 0.5 * float(r @ r) + 0.5 * reg * float(p @ p)
        history.append(J)
        # FD gradient
        g = np.zeros_like(p)
        i = 0
        while i < p.size:
            pp = p.copy()
            pp[i] += eps
            yp = forward(pp)
            g[i] = ((yp - target) @ (yp - target) - r @ r) / (2 * eps)
            i += 1
        g += reg * p
        p = p - lr * g
        if bounds is not None:
            lo, hi = bounds
            if lo is not None:
                p = np.maximum(p, lo)
            if hi is not None:
                p = np.minimum(p, hi)
        it += 1
    return p, history


__all__ = ["inverse_design"]
