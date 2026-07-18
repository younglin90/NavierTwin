"""Code verification — fit log-log slope of error vs h.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.verification.loglog_slope import slope_fit
    >>> h = np.array([0.1, 0.05, 0.025])
    >>> err = h ** 2
    >>> abs(slope_fit(h, err) - 2.0) < 1e-9
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def slope_fit(h: NDArray[np.float64], err: NDArray[np.float64]) -> float:
    """Linear fit log(err) vs log(h); slope = order p."""
    h = np.asarray(h, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    A = np.vstack([np.log(h), np.ones_like(h)]).T
    p, _ = np.linalg.lstsq(A, np.log(err + 1e-30), rcond=None)[0]
    return float(p)


__all__ = ["slope_fit"]
