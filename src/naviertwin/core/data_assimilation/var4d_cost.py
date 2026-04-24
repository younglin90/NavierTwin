"""4D-Var cost function — 배경+관측 항, FD gradient.

J(x0) = ½(x0 - xb)ᵀ B⁻¹ (x0 - xb) + ½ Σ (H(M_t x0) - y_t)ᵀ R⁻¹ (H(M_t x0) - y_t)

M: 모델 연산 (함수), H: 관측 연산, B/R: 공분산.

Examples:
    >>> # 사용은 테스트 참조
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def var4d_cost(
    x0: NDArray[np.float64],
    xb: NDArray[np.float64],
    B_inv: NDArray[np.float64],
    M: Callable[[NDArray, int], NDArray],
    H: Callable[[NDArray], NDArray],
    observations: list[NDArray[np.float64]],
    R_inv: NDArray[np.float64],
) -> float:
    """4D-Var 비용."""
    diff = x0 - xb
    J = 0.5 * float(diff @ B_inv @ diff)
    x = x0.copy()
    for t, y in enumerate(observations):
        x = M(x, t)
        innov = H(x) - y
        J += 0.5 * float(innov @ R_inv @ innov)
    return J


def fd_gradient(
    x0: NDArray[np.float64],
    cost_fn: Callable[[NDArray], float],
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    """비용함수 x0 에 대한 FD gradient."""
    x0 = np.asarray(x0, dtype=np.float64).ravel().copy()
    g = np.zeros_like(x0)
    J0 = cost_fn(x0)
    for i in range(x0.size):
        xp = x0.copy()
        xp[i] += eps
        g[i] = (cost_fn(xp) - J0) / eps
    return g


__all__ = ["var4d_cost", "fd_gradient"]
