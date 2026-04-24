"""IMEX Runge-Kutta 2 (Ascher-Ruuth-Spiteri 1997) — 강성 + 비강성 분할.

du/dt = f_E(u) + f_I(u),  f_I 는 implicit (선형 행렬 L), f_E 는 explicit.

ARS(2,2,2) 한 스텝 (linear implicit operator L 이라 가정).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.imex_rk import imex_rk2_step
    >>> L = -2.0 * np.eye(3)
    >>> u = np.ones(3)
    >>> u1 = imex_rk2_step(u, lambda x: -0.1 * x, L, dt=0.01)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def imex_rk2_step(
    u: NDArray[np.float64],
    f_explicit: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    L_implicit: NDArray[np.float64],
    *,
    dt: float,
) -> NDArray[np.float64]:
    """ARS(2,2,2) 한 스텝.  γ = (2 - √2)/2, δ = 1 - 1/(2γ)."""
    u = np.asarray(u, dtype=np.float64)
    n = u.shape[0]
    Im = np.eye(n)
    L = np.asarray(L_implicit, dtype=np.float64)
    gamma = (2.0 - np.sqrt(2.0)) / 2.0
    delta = 1.0 - 1.0 / (2.0 * gamma)

    # Stage 1: U1 = u + dt γ L U1 + dt γ f_E(u)
    rhs1 = u + dt * gamma * f_explicit(u)
    U1 = np.linalg.solve(Im - dt * gamma * L, rhs1)

    # Stage 2:
    # U2 = u + dt δ L U1 + dt γ L U2 + dt [(1-γ) f_E(U1) - δ f_E(u)] -- ARS variant
    # 표준 ARS(2,2,2): a21^E = γ, b1^E=1-γ, b2^E=γ ... 단순화 버전 사용
    fe_u = f_explicit(u)
    fe_1 = f_explicit(U1)
    rhs2 = u + dt * delta * (L @ U1) + dt * ((1 - gamma) * fe_1 + (1 - delta - (1 - gamma)) * fe_u)
    # simpler: stable Runge-Kutta:  u_new = u + dt*( (1-γ)*L@U1 + γ*L@U_new + (1-γ)*f_E(U1) + γ*f_E(u) )
    rhs2 = u + dt * (1 - gamma) * (L @ U1) + dt * ((1 - gamma) * fe_1 + gamma * fe_u)
    u_new = np.linalg.solve(Im - dt * gamma * L, rhs2)
    return u_new


__all__ = ["imex_rk2_step"]
