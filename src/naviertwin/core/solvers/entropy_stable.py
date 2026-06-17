"""Entropy-stable KEP flux (Kennedy-Gruber), compressible Euler.

간단 form: F = (ρ̄ ū, ρ̄ ū² + p̄, ρ̄ ū H̄).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.entropy_stable import kep_flux
    >>> UL = np.array([1.0, 0.5, 2.5])
    >>> UR = np.array([0.8, 0.4, 2.0])
    >>> F = kep_flux(UL, UR, gamma=1.4)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by entropy-stable fluxes")


def kep_flux(
    UL: NDArray, UR: NDArray, *, gamma: float = 1.4,
) -> NDArray:
    return _kernels.kep_flux(
        np.asarray(UL, dtype=np.float64),
        np.asarray(UR, dtype=np.float64),
        float(gamma),
    )


__all__ = ["kep_flux"]
