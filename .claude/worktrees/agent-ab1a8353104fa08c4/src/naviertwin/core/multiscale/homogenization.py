"""Asymptotic homogenization 1D — effective conductivity.

For a(y) periodic on (0,1): k_eff = (∫ 1/a dy)^-1 (harmonic mean).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.multiscale.homogenization import effective_conductivity_1d
    >>> a = np.array([1.0, 1.0, 1.0])
    >>> effective_conductivity_1d(a)
    1.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def effective_conductivity_1d(a_cell: NDArray[np.float64]) -> float:
    """Harmonic mean of cell conductivities."""
    a = np.asarray(a_cell, dtype=np.float64)
    return float(len(a) / np.sum(1.0 / (a + 1e-30)))


def cell_problem_solution(
    a_cell: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve d/dy(a (1 + dχ/dy)) = 0 with periodic BC; returns χ at cell centers."""
    a = np.asarray(a_cell, dtype=np.float64)
    n = len(a)
    # χ' = c/a - 1; integrate, enforce periodicity → c = harmonic mean
    c = effective_conductivity_1d(a)
    dy = 1.0 / n
    chi_prime = c / a - 1.0
    chi = np.cumsum(chi_prime) * dy
    chi -= chi.mean()
    return chi


__all__ = ["cell_problem_solution", "effective_conductivity_1d"]
