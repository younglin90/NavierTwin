"""Heterogeneous Multiscale Method — micro estimator at macro point.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.multiscale.hmm import hmm_macro_flux
    >>> def micro(u_macro): return 2.0 * u_macro
    >>> hmm_macro_flux(np.array([1.0, 2.0]), micro)
    array([2., 4.])
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def hmm_macro_flux(
    macro_state: NDArray[np.float64],
    micro_solver: Callable[[float], float],
) -> NDArray[np.float64]:
    """At each macro DOF, evaluate micro_solver and return result."""
    u = np.asarray(macro_state, dtype=np.float64)
    values = map(lambda ui: float(micro_solver(float(ui))), u)
    return np.fromiter(values, dtype=float, count=u.size)


__all__ = ["hmm_macro_flux"]
