"""Cut-cell Cartesian — fluid volume fraction from SDF (1D/2D).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.geometry.cut_cell import cut_cell_fraction_2d
    >>> phi = np.array([[-1, -1, 1], [-1, 0.5, 1], [1, 1, 1]], dtype=float)
    >>> frac = cut_cell_fraction_2d(phi)
    >>> frac.shape
    (2, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cut_cell_fraction_2d(phi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Interpolate each cell fraction from corner phi values.

    fraction = mean(corners < 0).  Approximate.
    """
    phi = np.asarray(phi, dtype=np.float64)
    corners = np.stack(
        [
            phi[:-1, :-1],
            phi[1:, :-1],
            phi[:-1, 1:],
            phi[1:, 1:],
        ],
        axis=0,
    )
    return (corners < 0).mean(axis=0)


__all__ = ["cut_cell_fraction_2d"]
