"""2D histogram helper — wrapper around np.histogram2d.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.visualization.hist2d import hist2d
    >>> rng = np.random.default_rng(0)
    >>> H, xe, ye = hist2d(rng.standard_normal(100), rng.standard_normal(100))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hist2d(
    x: NDArray[np.float64], y: NDArray[np.float64], *,
    bins: int = 30, density: bool = False,
) -> tuple[NDArray, NDArray, NDArray]:
    H, xe, ye = np.histogram2d(np.asarray(x), np.asarray(y), bins=bins, density=density)
    return H, xe, ye


__all__ = ["hist2d"]
