"""ROM / 상미분방정식 시간 적분 — Euler / RK2 / RK4 / adaptive RK45.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.time_integrator import integrate_ode
    >>> f = lambda t, y: -y
    >>> t, y = integrate_ode(f, y0=np.array([1.0]), t_span=(0, 1), dt=0.01, method="rk4")
    >>> abs(y[-1, 0] - np.exp(-1)) < 1e-6
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

RHS = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]


def _euler(f: RHS, t: float, y: NDArray, dt: float) -> NDArray:
    return y + dt * f(t, y)


def _rk2(f: RHS, t: float, y: NDArray, dt: float) -> NDArray:
    k1 = f(t, y)
    k2 = f(t + dt, y + dt * k1)
    return y + dt * 0.5 * (k1 + k2)


def _rk4(f: RHS, t: float, y: NDArray, dt: float) -> NDArray:
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


_STEPPERS = {"euler": _euler, "rk2": _rk2, "rk4": _rk4}


def integrate_ode(
    f: RHS,
    y0: NDArray[np.float64],
    t_span: tuple[float, float],
    dt: float,
    method: str = "rk4",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """고정 스텝 적분. (times, states[n_steps+1, dim]) 반환."""
    if method not in _STEPPERS:
        raise ValueError(f"method ∈ {list(_STEPPERS)}")
    step = _STEPPERS[method]
    t0, tf = t_span
    n = int(np.ceil((tf - t0) / dt))
    ts = np.linspace(t0, t0 + n * dt, n + 1)
    y = np.asarray(y0, dtype=np.float64).copy()
    ys = np.zeros((n + 1, y.size), dtype=np.float64)
    ys[0] = y
    i = 0
    while i < n:
        y = step(f, ts[i], y, dt)
        ys[i + 1] = y
        i += 1
    return ts, ys


def integrate_scipy(
    f: RHS,
    y0: NDArray[np.float64],
    t_span: tuple[float, float],
    *,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    t_eval: NDArray | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """scipy.integrate.solve_ivp 래퍼 — adaptive."""
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:
        raise RuntimeError("scipy 필요") from exc
    sol = solve_ivp(
        f, t_span, np.asarray(y0, dtype=np.float64),
        method=method, rtol=rtol, atol=atol, t_eval=t_eval,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp 실패: {sol.message}")
    return sol.t, sol.y.T


__all__ = ["integrate_ode", "integrate_scipy"]
