"""FSI 1-way — fluid pressure → solid surface load mapping.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.coupling.fsi_oneway import map_pressure_to_nodes
    >>> p = np.array([1.0, 2.0])
    >>> nrm = np.array([[0., 1, 0], [0., 1, 0]])
    >>> w = np.array([0.5, 0.5])
    >>> F = map_pressure_to_nodes(p, nrm, w)
    >>> F.shape
    (2, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def map_pressure_to_nodes(
    pressure: NDArray[np.float64],
    normals: NDArray[np.float64],
    areas: NDArray[np.float64],
) -> NDArray[np.float64]:
    """F_i = -p_i n_i A_i (forces per face)."""
    p = np.asarray(pressure, dtype=np.float64).ravel()
    n = np.asarray(normals, dtype=np.float64)
    a = np.asarray(areas, dtype=np.float64).ravel()
    return -(p * a)[:, None] * n


__all__ = ["map_pressure_to_nodes"]
