"""Method of Manufactured Solutions — generate source term S(x) = L u_manuf.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.verification.mms import poisson_1d_source
    >>> x = np.linspace(0, 1, 11)
    >>> S = poisson_1d_source(x, lambda x: np.sin(np.pi*x))
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def poisson_1d_source(
    x: NDArray[np.float64],
    u_manuf: Callable[[NDArray], NDArray],
    *,
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """For -u'' = S, return S via FD second derivative."""
    x = np.asarray(x, dtype=np.float64)
    u = u_manuf(x)
    upp = (u_manuf(x + eps) - 2 * u + u_manuf(x - eps)) / (eps * eps)
    return -upp


def l2_error(
    u_num: NDArray[np.float64], u_exact: NDArray[np.float64], *, dx: float = 1.0,
) -> float:
    return float(np.sqrt(np.sum((u_num - u_exact) ** 2) * dx))


__all__ = ["l2_error", "poisson_1d_source"]
