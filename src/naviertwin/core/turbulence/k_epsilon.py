"""k-ε closure model — 간단 2-eq RANS.

    ∂_t k   = P_k - ε + ∇·(ν_t/σ_k · ∇k)
    ∂_t ε   = C_ε1 · ε/k · P_k - C_ε2 · ε²/k + ∇·(ν_t/σ_ε · ∇ε)
    ν_t = C_μ · k² / ε

여기서 P_k = ν_t · |S|², S = 0.5·(∇u + ∇uᵀ) (RANS 평균 장 속도 구배).

2D 단일 스텝 업데이트 함수 — 학습용/데모용.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.turbulence.k_epsilon import eddy_viscosity
    >>> k = 0.5 * np.ones((10, 10))
    >>> eps = 0.1 * np.ones((10, 10))
    >>> nu_t = eddy_viscosity(k, eps)
    >>> nu_t.shape
    (10, 10)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")

# 표준 k-ε 계수
C_MU = 0.09
C_EPS1 = 1.44
C_EPS2 = 1.92
SIGMA_K = 1.0
SIGMA_EPS = 1.3


def eddy_viscosity(
    k: NDArray[np.float64],
    epsilon: NDArray[np.float64],
) -> NDArray[np.float64]:
    """ν_t = C_μ · k² / ε."""
    k = np.asarray(k, dtype=np.float64)
    eps = np.asarray(epsilon, dtype=np.float64)
    if np.any(k < 0) or np.any(eps < 0):
        raise ValueError("k, ε 는 비음 여야 합니다")
    return C_MU * k ** 2 / np.maximum(eps, 1e-12)


def production_rate(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    dx: float,
    dy: float,
    nu_t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """P_k = 2 ν_t · (e_ij e_ij).

    e_ij = 0.5·(∂_i u_j + ∂_j u_i).
    """
    return np.asarray(
        _kernels.production_rate_2d(
            np.asarray(u, dtype=np.float64),
            np.asarray(v, dtype=np.float64),
            float(dx),
            float(dy),
            np.asarray(nu_t, dtype=np.float64),
        ),
        dtype=np.float64,
    )


def k_epsilon_step(
    k: NDArray[np.float64],
    epsilon: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    dt: float,
    dx: float,
    dy: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """k-ε 한 스텝 업데이트 (explicit).

    Returns:
        (k_new, ε_new).
    """
    nu_t = eddy_viscosity(k, epsilon)
    P_k = production_rate(u, v, dx, dy, nu_t)

    # 확산항 (laplacian)
    def lap(a: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            (np.roll(a, 1, axis=0) - 2 * a + np.roll(a, -1, axis=0)) / dy ** 2
            + (np.roll(a, 1, axis=1) - 2 * a + np.roll(a, -1, axis=1)) / dx ** 2
        )

    dk = P_k - epsilon + (nu_t / SIGMA_K) * lap(k)
    deps = (
        C_EPS1 * epsilon / np.maximum(k, 1e-12) * P_k
        - C_EPS2 * epsilon ** 2 / np.maximum(k, 1e-12)
        + (nu_t / SIGMA_EPS) * lap(epsilon)
    )
    k_new = np.maximum(k + dt * dk, 1e-12)
    eps_new = np.maximum(epsilon + dt * deps, 1e-12)
    return k_new, eps_new


__all__ = ["eddy_viscosity", "production_rate", "k_epsilon_step"]
