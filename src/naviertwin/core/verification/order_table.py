"""Order-of-accuracy table — pairwise log ratio.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.verification.order_table import order_table
    >>> h = np.array([0.1, 0.05, 0.025])
    >>> err = h ** 2
    >>> table = order_table(h, err)
    >>> len(table['p_pair'])
    2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def order_table(
    h: NDArray[np.float64], err: NDArray[np.float64],
) -> dict:
    return dict(
        _kernels.order_table(
            np.asarray(h, dtype=np.float64),
            np.asarray(err, dtype=np.float64),
        ),
    )


__all__ = ["order_table"]
