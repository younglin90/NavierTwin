"""Reynolds decomposition + fluctuation statistics.

u = ū + u', v = v̄ + v', Reynolds stresses <u'v'>, TKE.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.reynolds import reynolds_decompose
    >>> U = np.random.default_rng(0).standard_normal((100, 50))  # T, N
    >>> mean, fluc = reynolds_decompose(U)
    >>> mean.shape, fluc.shape
    ((50,), (100, 50))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def reynolds_decompose(
    U: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(T, N) 시계열 → 시간평균 ū(N,) + 변동 u'(T, N)."""
    U = np.asarray(U, dtype=np.float64)
    if U.ndim != 2:
        raise ValueError("U (T, N)")
    m = U.mean(axis=0)
    return m, U - m


def reynolds_stress(
    u_fluc: NDArray[np.float64], v_fluc: NDArray[np.float64],
) -> NDArray[np.float64]:
    """<u'v'> 시간 평균 (N,)."""
    return (np.asarray(u_fluc) * np.asarray(v_fluc)).mean(axis=0)


def turbulence_intensity(U: NDArray[np.float64]) -> NDArray[np.float64]:
    """I = u'_rms / |ū| 포인트별."""
    m, fluc = reynolds_decompose(U)
    rms = np.sqrt((fluc ** 2).mean(axis=0))
    return rms / (np.abs(m) + 1e-30)


def tke_pointwise(
    U: NDArray[np.float64], V: NDArray[np.float64],
    W: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """k(N) = ½ <u'² + v'² (+ w'²)>."""
    _, uf = reynolds_decompose(U)
    _, vf = reynolds_decompose(V)
    s = (uf ** 2).mean(axis=0) + (vf ** 2).mean(axis=0)
    if W is not None:
        _, wf = reynolds_decompose(W)
        s = s + (wf ** 2).mean(axis=0)
    return 0.5 * s


__all__ = [
    "reynolds_decompose", "reynolds_stress", "turbulence_intensity",
    "tke_pointwise",
]
