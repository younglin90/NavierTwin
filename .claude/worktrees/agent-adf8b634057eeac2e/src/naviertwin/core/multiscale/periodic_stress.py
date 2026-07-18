"""Periodic-cell stress homogenization — volume-averaged Cauchy stress.

⟨σ⟩ = (1/V) ∫ σ dV.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.multiscale.periodic_stress import volume_average_stress
    >>> sigma = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    >>> avg = volume_average_stress(np.tile(sigma, (5, 1, 1)), volumes=np.ones(5))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def volume_average_stress(
    sigma: NDArray[np.float64],
    volumes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """sigma shape (N, 3, 3); volumes shape (N,)."""
    s = np.asarray(sigma, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)
    return np.einsum("nij,n->ij", s, v) / v.sum()


def hill_mandel_check(
    macro_stress: NDArray[np.float64],
    macro_strain: NDArray[np.float64],
    micro_work: float,
    *,
    tol: float = 1e-6,
) -> bool:
    """⟨σ : ε⟩ ≈ ⟨σ⟩ : ⟨ε⟩."""
    macro_work = float(np.tensordot(macro_stress, macro_strain))
    return abs(macro_work - micro_work) < tol


__all__ = ["hill_mandel_check", "volume_average_stress"]
