"""Gradient confusion — Sankararaman et al. 2020.

ζ = (1/(B(B-1))) Σ_{i≠j} cos(g_i, g_j); negative → adversarial gradients.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.grad_confusion import grad_confusion
    >>> g = np.eye(3)
    >>> grad_confusion(g)
    0.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def grad_confusion(grads: NDArray[np.float64]) -> float:
    """grads: (B, d) per-sample gradients."""
    g = np.asarray(grads, dtype=np.float64)
    norms = np.linalg.norm(g, axis=1, keepdims=True) + 1e-30
    g_n = g / norms
    cos = g_n @ g_n.T
    B = len(g)
    if B < 2:
        return 0.0
    off_diag = (cos.sum() - np.trace(cos)) / (B * (B - 1))
    return float(off_diag)


__all__ = ["grad_confusion"]
