"""Nonlinear Kalman smoother — RTS-style with EKF linearization.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_assimilation.ks_nonlinear import nonlinear_rts
    >>> xs = np.zeros((5, 2)); Ps = np.tile(np.eye(2), (5, 1, 1))
    >>> A = np.eye(2)
    >>> xs_s, Ps_s = nonlinear_rts(xs, Ps, A, Q=0.01*np.eye(2))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.data_assimilation.iterated_ekf import _right_solve


def nonlinear_rts(
    xs_filt: NDArray[np.float64],
    Ps_filt: NDArray[np.float64],
    F: NDArray[np.float64],
    *,
    Q: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """RTS backward pass given EKF-filtered means/covs and linearized F."""
    xs = np.asarray(xs_filt, dtype=np.float64).copy()
    Ps = np.asarray(Ps_filt, dtype=np.float64).copy()
    N = xs.shape[0]
    k = N - 2
    while k >= 0:
        Pp = F @ Ps[k] @ F.T + Q
        Ck = _right_solve(Pp, Ps[k] @ F.T)
        xs[k] = xs[k] + Ck @ (xs[k + 1] - F @ xs[k])
        Ps[k] = Ps[k] + Ck @ (Ps[k + 1] - Pp) @ Ck.T
        k -= 1
    return xs, Ps


__all__ = ["nonlinear_rts"]
