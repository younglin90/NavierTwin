"""Coarse-graining — block-average and moving-window filter.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.multiscale.coarse_grain import block_average
    >>> block_average(np.arange(8.0), block=2)
    array([0.5, 2.5, 4.5, 6.5])
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def block_average(x: NDArray[np.float64], block: int) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    n = (len(x) // block) * block
    return x[:n].reshape(-1, block).mean(axis=1)


def moving_average(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return (cs[window:] - cs[:-window]) / window


__all__ = ["block_average", "moving_average"]
