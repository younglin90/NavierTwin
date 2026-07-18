"""내장 1D PDE 참조 솔버 — Burgers + Heat.

ROM/Operator 학습용 합성 데이터 생성에 사용.

Burgers:
    u_t + u u_x = ν u_xx,   주기경계, 초기조건 u0.

Heat:
    u_t = α u_xx,   Dirichlet u(0,t)=u(L,t)=0.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solver_interfaces.pde_solvers import solve_burgers_1d
    >>> u0 = np.sin(np.linspace(0, 2 * np.pi, 64))
    >>> t, U = solve_burgers_1d(u0, nu=0.01, L=2 * np.pi, T=0.5, n_steps=200)
    >>> U.shape[0] == 200 and U.shape[1] == 64
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def solve_burgers_1d(
    u0: NDArray[np.float64],
    nu: float = 0.01,
    L: float = 2 * np.pi,
    T: float = 1.0,
    n_steps: int = 500,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """1D Burgers 주기경계 FD explicit/semi-implicit.

    - 이류항: upwind 1차
    - 확산항: 중심차분
    - 시간: explicit forward Euler (CFL 조건 필요)

    Returns:
        (times, U) — U shape = (n_steps, N).
    """
    u0 = np.asarray(u0, dtype=np.float64)
    N = u0.size
    dx = L / N
    dt = T / n_steps

    if dt > 0.5 * dx ** 2 / max(nu, 1e-12):
        logger.warning(
            "Burgers CFL 위험: dt=%.3g > 0.5·dx²/ν=%.3g", dt, 0.5 * dx ** 2 / max(nu, 1e-12),
        )

    times = np.linspace(dt, T, n_steps)
    U = np.zeros((n_steps, N), dtype=np.float64)
    u = u0.copy()
    k = 0
    while k < n_steps:
        # 주기경계: np.roll
        du_plus = u - np.roll(u, 1)     # positive-velocity upwind backward
        du_minus = np.roll(u, -1) - u   # negative-velocity forward
        adv = np.where(u > 0, u * du_plus / dx, u * du_minus / dx)
        diff = nu * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx ** 2
        u = u + dt * (-adv + diff)
        U[k] = u
        k += 1
    return times, U


def solve_heat_1d(
    u0: NDArray[np.float64],
    alpha: float = 0.01,
    L: float = 1.0,
    T: float = 1.0,
    n_steps: int = 500,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """1D heat equation — Dirichlet 0 boundary, Crank-Nicolson.

    Returns:
        (times, U) — U shape = (n_steps, N+1) (경계 포함).
    """
    u0 = np.asarray(u0, dtype=np.float64).copy()
    N = u0.size - 1
    dx = L / N
    dt = T / n_steps
    r = alpha * dt / (2 * dx ** 2)

    # Tridiagonal matrices used by Crank-Nicolson
    # Boundary 노드 (0, N) 고정 0 → interior 만 풀이
    interior = N - 1
    main_A = (1 + 2 * r) * np.ones(interior)
    off_A = -r * np.ones(interior - 1)
    main_B = (1 - 2 * r) * np.ones(interior)
    off_B = r * np.ones(interior - 1)

    def _tridiag_solve(
        main: NDArray[np.float64], off: NDArray[np.float64], rhs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Thomas algorithm
        n = main.size
        m = main.copy()
        r_ = rhs.copy()
        i = 1
        while i < n:
            w = off[i - 1] / m[i - 1]
            m[i] -= w * off[i - 1]
            r_[i] -= w * r_[i - 1]
            i += 1
        x = np.zeros(n)
        x[-1] = r_[-1] / m[-1]
        i = n - 2
        while i >= 0:
            x[i] = (r_[i] - off[i] * x[i + 1]) / m[i]
            i -= 1
        return x

    times = np.linspace(dt, T, n_steps)
    U = np.zeros((n_steps, N + 1), dtype=np.float64)
    u = u0.copy()
    u[0] = 0.0
    u[-1] = 0.0
    k = 0
    while k < n_steps:
        # RHS = B u_interior
        interior_u = u[1:-1]
        rhs = main_B * interior_u
        rhs[:-1] += off_B * interior_u[1:]
        rhs[1:] += off_B * interior_u[:-1]
        # 경계값 0 이므로 boundary 기여 없음
        u_new_interior = _tridiag_solve(main_A, off_A, rhs)
        u = np.concatenate([[0.0], u_new_interior, [0.0]])
        U[k] = u
        k += 1
    return times, U


__all__ = ["solve_burgers_1d", "solve_heat_1d"]
