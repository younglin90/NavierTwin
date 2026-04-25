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


def order_table(
    h: NDArray[np.float64], err: NDArray[np.float64],
) -> dict:
    h = np.asarray(h, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    p_pair = []
    for i in range(len(h) - 1):
        p = float(np.log(err[i] / err[i + 1]) / np.log(h[i] / h[i + 1]))
        p_pair.append(p)
    return {"h": h.tolist(), "err": err.tolist(), "p_pair": p_pair}


__all__ = ["order_table"]
