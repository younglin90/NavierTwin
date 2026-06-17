"""FE² (Feyel) two-scale skeleton — macro element invokes micro RVE.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.multiscale.fe2 import fe2_macro_stress
    >>> def micro(strain): return 200.0 * strain
    >>> fe2_macro_stress(np.array([0.01, 0.02]), micro)
    array([2., 4.])
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def fe2_macro_stress(
    macro_strain: NDArray[np.float64],
    micro_rve: Callable[[float], float],
) -> NDArray[np.float64]:
    s = np.asarray(macro_strain, dtype=np.float64)
    return np.fromiter(map(lambda e: float(micro_rve(float(e))), s), dtype=np.float64, count=s.size)


def fe2_tangent_fd(
    macro_strain: NDArray[np.float64],
    micro_rve: Callable[[float], float],
    *,
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    s = np.asarray(macro_strain, dtype=np.float64)
    plus = map(lambda e: float(micro_rve(float(e) + eps)), s)
    minus = map(lambda e: float(micro_rve(float(e) - eps)), s)
    return (np.fromiter(plus, dtype=np.float64, count=s.size) - np.fromiter(minus, dtype=np.float64, count=s.size)) / (
        2 * eps
    )


__all__ = ["fe2_macro_stress", "fe2_tangent_fd"]
