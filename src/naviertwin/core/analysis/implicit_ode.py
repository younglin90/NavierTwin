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

from naviertwin._native import HAS_NATIVE_KERNELS, _kernels


def _solve_dense(A: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    mat = np.ascontiguousarray(A, dtype=np.float64)
    rhs = np.ascontiguousarray(b, dtype=np.float64)
    if HAS_NATIVE_KERNELS and _kernels is not None:
        try:
            return _kernels.solve_dense(mat, rhs)
        except Exception:
            pass
    return getattr(np.linalg, "solve")(mat, rhs)


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
    ys = np.zeros((n + 1, d))
    ys[0] = y0
    y = y0.copy()
    k = 0
    while k < n:
        rhs = y + dt * bb
        y = _solve_dense(M, rhs)
        ys[k + 1] = y
        k += 1
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
    k = 0
    while k < n:
        rhs = R @ y + dt * bb
        y = _solve_dense(L, rhs)
        ys[k + 1] = y
        k += 1
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
    k = 0
    while k < n:
        tnp1 = ts[k + 1]
        yn = y.copy()
        # fixed-point init
        z = yn + dt * f(ts[k], yn)
        newton_idx = 0
        while newton_idx < newton_iters:
            r = z - yn - dt * f(tnp1, z)
            if np.linalg.norm(r) < tol:
                break
            # FD Jacobian of g(z) = z - yn - dt f(t, z)
            J = eye.copy()
            eps = 1e-6
            fz = f(tnp1, z)
            j = 0
            while j < d:
                zp = z.copy()
                zp[j] += eps
                J[:, j] -= dt * (f(tnp1, zp) - fz) / eps
                j += 1
            dz = _solve_dense(J, -r)
            z = z + dz
            newton_idx += 1
        y = z
        ys[k + 1] = y
        k += 1
    return ts, ys


__all__ = [
    "implicit_euler_linear", "crank_nicolson_linear", "implicit_euler_nonlinear",
]
