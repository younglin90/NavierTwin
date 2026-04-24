"""Conjugate Heat Transfer — fluid-solid interface coupling (Dirichlet/Neumann).

1D 모델: solid (k_s) ↔ fluid (k_f), 인터페이스 T 와 q 매칭.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.coupling.cht import cht_iterate
    >>> Ts = np.linspace(300, 350, 5)
    >>> Tf = np.linspace(400, 350, 5)
    >>> Ts2, Tf2 = cht_iterate(Ts, Tf, k_s=10.0, k_f=1.0, n_iter=20)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cht_iterate(
    T_solid: NDArray[np.float64],
    T_fluid: NDArray[np.float64],
    *,
    k_s: float = 10.0,
    k_f: float = 1.0,
    n_iter: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Dirichlet-Neumann 반복: 인터페이스 (마지막/첫 노드) 매칭."""
    Ts = np.asarray(T_solid, dtype=np.float64).copy()
    Tf = np.asarray(T_fluid, dtype=np.float64).copy()
    for _ in range(n_iter):
        # solid: Laplace, fluid edge T_solid[-1] passed as Dirichlet to fluid[0]
        # solid solve (1D Laplace, both ends fixed)
        Ts[1:-1] = 0.5 * (Ts[2:] + Ts[:-2])
        Tf[1:-1] = 0.5 * (Tf[2:] + Tf[:-2])
        # heat-flux match at interface (Ts[-1] ↔ Tf[0])
        T_iface = (k_s * Ts[-2] + k_f * Tf[1]) / (k_s + k_f)
        Ts[-1] = T_iface
        Tf[0] = T_iface
    return Ts, Tf


__all__ = ["cht_iterate"]
