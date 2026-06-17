"""Symplectic integrators — energy-preserving Hamiltonian steppers.

Leapfrog + Velocity Verlet. H(p, q) = T(p) + V(q), 보존성이 비보존형 RK4 보다 좋음.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.symplectic import velocity_verlet
    >>> # 1D 조화진동자 ẍ = -x, m=1, k=1
    >>> def force(q): return -q
    >>> q, p = np.array([1.0]), np.array([0.0])
    >>> qs, ps = velocity_verlet(force, q, p, dt=0.01, n=1000)
    >>> abs(0.5*(qs[-1,0]**2 + ps[-1,0]**2) - 0.5) < 1e-4
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

Force = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def velocity_verlet(
    force_fn: Force,
    q0: NDArray[np.float64], p0: NDArray[np.float64],
    *, dt: float, n: int, mass: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Velocity Verlet. F(q) = -dV/dq."""
    q = np.asarray(q0, dtype=np.float64).copy()
    p = np.asarray(p0, dtype=np.float64).copy()
    qs = np.zeros((n + 1, q.size))
    ps = np.zeros((n + 1, p.size))
    qs[0] = q
    ps[0] = p
    F = force_fn(q)
    k = 0
    while k < n:
        p_half = p + 0.5 * dt * F
        q = q + dt * p_half / mass
        F = force_fn(q)
        p = p_half + 0.5 * dt * F
        qs[k + 1] = q
        ps[k + 1] = p
        k += 1
    return qs, ps


def leapfrog(
    force_fn: Force,
    q0: NDArray[np.float64], p0: NDArray[np.float64],
    *, dt: float, n: int, mass: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Position Verlet / leapfrog."""
    q = np.asarray(q0, dtype=np.float64).copy()
    p = np.asarray(p0, dtype=np.float64).copy()
    qs = np.zeros((n + 1, q.size))
    ps = np.zeros((n + 1, p.size))
    qs[0] = q
    ps[0] = p
    k = 0
    while k < n:
        q_half = q + 0.5 * dt * p / mass
        p = p + dt * force_fn(q_half)
        q = q_half + 0.5 * dt * p / mass
        qs[k + 1] = q
        ps[k + 1] = p
        k += 1
    return qs, ps


__all__ = ["velocity_verlet", "leapfrog"]
