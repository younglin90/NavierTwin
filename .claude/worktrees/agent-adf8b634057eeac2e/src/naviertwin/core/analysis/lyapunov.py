"""Largest Lyapunov exponent — Wolf 1985 스타일 1차원 맵.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.lyapunov import lyapunov_map
    >>> # logistic r=4 → λ = ln(2) ≈ 0.693
    >>> f = lambda x: 4 * x * (1 - x)
    >>> fp = lambda x: 4 - 8 * x
    >>> l = lyapunov_map(f, fp, x0=0.4, n=5000)
    >>> abs(l - np.log(2)) < 0.05
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def lyapunov_map(
    f: Callable[[float], float],
    fprime: Callable[[float], float],
    x0: float, n: int = 10000, *, warmup: int = 100,
) -> float:
    """1D map 의 LLE = <log|f'(x_k)|>."""
    x = float(x0)
    warmup_idx = 0
    while warmup_idx < warmup:
        x = f(x)
        warmup_idx += 1
    s = 0.0
    step_idx = 0
    while step_idx < n:
        s += np.log(abs(fprime(x)) + 1e-30)
        x = f(x)
        step_idx += 1
    return float(s / n)


def benettin_flow(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray, t_end: float, dt: float,
    *, renorm_every: int = 10,
) -> float:
    """연속 ODE 의 LLE — Benettin 알고리즘 (유한 스텝 RK4 + 재정규화)."""
    y = np.asarray(y0, dtype=np.float64).copy()
    d = y.size
    q = np.eye(d, 1).ravel()  # 초기 perturbation
    # normalize
    q = q / (np.linalg.norm(q) + 1e-30)

    def rk4(y, t):
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs(t + dt, y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    s = 0.0
    steps = int(np.ceil(t_end / dt))
    t = 0.0
    k = 0
    while k < steps:
        J = jac(t, y)
        # linear evolution of perturbation: q += dt J q (RK4 도 가능, 여기서는 Euler)
        q = q + dt * J @ q
        y = rk4(y, t)
        t += dt
        if (k + 1) % renorm_every == 0:
            norm = np.linalg.norm(q)
            s += np.log(norm + 1e-30)
            q = q / (norm + 1e-30)
        k += 1
    return float(s / (steps * dt))


__all__ = ["lyapunov_map", "benettin_flow"]
