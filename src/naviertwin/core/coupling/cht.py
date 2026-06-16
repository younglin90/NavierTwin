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

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by CHT coupling")


def cht_iterate(
    T_solid: NDArray[np.float64],
    T_fluid: NDArray[np.float64],
    *,
    k_s: float = 10.0,
    k_f: float = 1.0,
    n_iter: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Dirichlet-Neumann 반복: 인터페이스 (마지막/첫 노드) 매칭."""
    Ts, Tf = _kernels.cht_iterate(
        np.asarray(T_solid, dtype=np.float64),
        np.asarray(T_fluid, dtype=np.float64),
        float(k_s),
        float(k_f),
        int(n_iter),
    )
    return np.asarray(Ts, dtype=np.float64), np.asarray(Tf, dtype=np.float64)


__all__ = ["cht_iterate"]
