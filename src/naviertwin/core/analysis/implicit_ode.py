"""경직 ODE 용 암시적 적분기 — Implicit Euler, Crank-Nicolson.

선형 시스템 y' = A y + b 에 최적 (직접해), 일반 비선형은 Newton 반복.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.implicit_ode import implicit_euler_linear
    >>> A = np.array([[-1000., 0.], [0., -1.]])  # stiff
    >>> y0 = np.array([1., 1.])
    >>> ts, ys = implicit_euler_linear(A, y0, t_span=(0, 1), dt=0.01)
    >>> abs(ys[-1, 0]) < 1e-4
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def implicit_euler_linear(
    A: NDArray[np.float64], y0: NDArray[np.float64],
    t_span: tuple[float, float], dt: float,
    b: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """y' = A y + b. (I - dt A) y_{n+1} = y_n + dt b."""
    n = int(np.ceil((t_span[1] - t_span[0]) / dt))
    dt = (t_span[1] - t_span[0]) / n
    ts = np.linspace(t_span[0], t_span[1], n + 1)
    d = y0.size
    bb = np.zeros(d) if b is None else np.asarray(b, dtype=np.float64)
    M = np.eye(d) - dt * np.asarray(A, dtype=np.float64)
    lu, piv = None, None
    try:
        from scipy.linalg import lu_factor, lu_solve

        lu, piv = lu_factor(M)
    except ImportError:
        pass
    ys = np.zeros((n + 1, d))
    ys[0] = y0
    y = y0.copy()
    for k in range(n):
        rhs = y + dt * bb
        if lu is not None:
            y = lu_solve((lu, piv), rhs)
        else:
            y = np.linalg.solve(M, rhs)
        ys[k + 1] = y
    return ts, ys


def crank_nicolson_linear(
    A: NDArray[np.float64], y0: NDArray[np.float64],
    t_span: tuple[float, float], dt: float,
    b: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(I - dt/2 A) y_{n+1} = (I + dt/2 A) y_n + dt b."""
    n = int(np.ceil((t_span[1] - t_span[0]) / dt))
    dt = (t_span[1] - t_span[0]) / n
    ts = np.linspace(t_span[0], t_span[1], n + 1)
    d = y0.size
    bb = np.zeros(d) if b is None else np.asarray(b, dtype=np.float64)
    Amat = np.asarray(A, dtype=np.float64)
    L = np.eye(d) - 0.5 * dt * Amat
    R = np.eye(d) + 0.5 * dt * Amat
    ys = np.zeros((n + 1, d))
    ys[0] = y0
    y = y0.copy()
    for k in range(n):
        rhs = R @ y + dt * bb
        y = np.linalg.solve(L, rhs)
        ys[k + 1] = y
    return ts, ys


def implicit_euler_nonlinear(
    f: Callable[[float, NDArray], NDArray], y0: NDArray,
    t_span: tuple[float, float], dt: float,
    *, newton_iters: int = 20, tol: float = 1e-10,
) -> tuple[NDArray, NDArray]:
    """y_{n+1} = y_n + dt f(t_{n+1}, y_{n+1}) via Newton iter (FD Jacobian)."""
    n = int(np.ceil((t_span[1] - t_span[0]) / dt))
    dt = (t_span[1] - t_span[0]) / n
    ts = np.linspace(t_span[0], t_span[1], n + 1)
    y0 = np.asarray(y0, dtype=np.float64)
    d = y0.size
    ys = np.zeros((n + 1, d))
    ys[0] = y0
    y = y0.copy()
    eye = np.eye(d)
    for k in range(n):
        tnp1 = ts[k + 1]
        yn = y.copy()
        # fixed-point init
        z = yn + dt * f(ts[k], yn)
        for _ in range(newton_iters):
            r = z - yn - dt * f(tnp1, z)
            if np.linalg.norm(r) < tol:
                break
            # FD Jacobian of g(z) = z - yn - dt f(t, z)
            J = eye.copy()
            eps = 1e-6
            for j in range(d):
                zp = z.copy()
                zp[j] += eps
                J[:, j] -= dt * (f(tnp1, zp) - f(tnp1, z)) / eps
            dz = np.linalg.solve(J, -r)
            z = z + dz
        y = z
        ys[k + 1] = y
    return ts, ys


__all__ = [
    "implicit_euler_linear", "crank_nicolson_linear", "implicit_euler_nonlinear",
]
