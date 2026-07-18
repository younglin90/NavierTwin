"""p-refinement — DG order bump per cell.

각 cell 의 polynomial order p_i 를 증가/감소시키는 정책.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.amr.p_refine import bump_order
    >>> errs = np.array([0.1, 0.5, 0.01, 0.3])
    >>> p = np.array([2, 2, 2, 2])
    >>> p2 = bump_order(p, errs, threshold_up=0.2, threshold_down=0.05)
    >>> p2.tolist()
    [2, 3, 1, 3]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bump_order(
    p: NDArray[np.int_],
    errors: NDArray[np.float64],
    *,
    threshold_up: float = 0.1,
    threshold_down: float = 0.01,
    p_min: int = 1,
    p_max: int = 6,
) -> NDArray[np.int_]:
    p = np.asarray(p, dtype=int).copy()
    e = np.asarray(errors, dtype=float)
    p[e > threshold_up] += 1
    p[e < threshold_down] -= 1
    return np.clip(p, p_min, p_max)


__all__ = ["bump_order"]
