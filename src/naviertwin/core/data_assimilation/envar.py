"""EnVar — Hybrid 3D-Var / ensemble (간단).

B = α B_static + (1-α) B_ensemble.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_assimilation.envar import envar_analysis
    >>> rng = np.random.default_rng(0)
    >>> xb = np.zeros(3)
    >>> ens = rng.standard_normal((10, 3))
    >>> z = np.array([1.0])
    >>> H = np.eye(3)[:1]
    >>> R = 0.1 * np.eye(1)
    >>> Bs = np.eye(3)
    >>> xa = envar_analysis(xb, ens, z, H, R, Bs, alpha=0.5)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def envar_analysis(
    xb: NDArray[np.float64],
    ensemble: NDArray[np.float64],
    z: NDArray[np.float64],
    H: NDArray[np.float64],
    R: NDArray[np.float64],
    B_static: NDArray[np.float64],
    *,
    alpha: float = 0.5,
) -> NDArray[np.float64]:
    """3D-Var with hybrid B."""
    ens = np.asarray(ensemble, dtype=np.float64)
    mean = ens.mean(axis=0)
    pert = ens - mean  # (N, n)
    B_ens = pert.T @ pert / max(len(ens) - 1, 1)
    B = alpha * B_static + (1 - alpha) * B_ens
    K = B @ H.T @ np.linalg.inv(H @ B @ H.T + R)
    return xb + K @ (z - H @ xb)


__all__ = ["envar_analysis"]
