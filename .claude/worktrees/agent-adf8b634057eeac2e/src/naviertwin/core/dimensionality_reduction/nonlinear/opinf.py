"""Operator Inference — Peherstorfer & Willcox 2016.

ẋ ≈ A x + H (x ⊗ x) + B u; project & fit reduced ops in latent.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.opinf import opinf_fit
    >>> rng = np.random.default_rng(0)
    >>> Z = rng.standard_normal((100, 3))
    >>> Zdot = -Z
    >>> ops = opinf_fit(Z, Zdot)
    >>> ops['A'].shape
    (3, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def opinf_fit(
    Z: NDArray[np.float64],
    Zdot: NDArray[np.float64],
    *,
    quadratic: bool = False,
    U: NDArray[np.float64] | None = None,
) -> dict:
    """Linear (and optional quadratic) operator inference."""
    Z = np.asarray(Z, dtype=np.float64)
    Zdot = np.asarray(Zdot, dtype=np.float64)
    T, r = Z.shape
    blocks = [Z]
    if quadratic:
        # Kronecker (unique pairs)
        kron = (Z[:, :, np.newaxis] * Z[:, np.newaxis, :]).reshape(T, r * r)
        blocks.append(kron)
    if U is not None:
        blocks.append(np.asarray(U).reshape(T, -1))
    Theta = np.hstack(blocks)
    Xi, *_ = np.linalg.lstsq(Theta, Zdot, rcond=None)
    ops = {}
    idx = 0
    ops["A"] = Xi[idx:idx + r, :].T
    idx += r
    if quadratic:
        ops["H"] = Xi[idx:idx + r * r, :].T
        idx += r * r
    if U is not None:
        nu = np.asarray(U).reshape(T, -1).shape[1]
        ops["B"] = Xi[idx:idx + nu, :].T
    return ops


__all__ = ["opinf_fit"]
