"""Helmholtz 분해 — 2D 주기경계 벡터장을 solenoidal + irrotational 로 분리.

    u = u_sol + u_irr
    ∇·u_sol = 0,    ∇×u_irr = 0

FFT 기반:
    û = û_sol + û_irr,  û_irr = k (k·û) / |k|²,  û_sol = û - û_irr.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.flow_analysis.helmholtz import helmholtz_2d
    >>> rng = np.random.default_rng(0)
    >>> u = rng.standard_normal((16, 16))
    >>> v = rng.standard_normal((16, 16))
    >>> u_s, v_s, u_i, v_i = helmholtz_2d(u, v)
    >>> np.allclose(u, u_s + u_i)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def helmholtz_2d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    Lx: float = 2 * np.pi,
    Ly: float = 2 * np.pi,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64],
    NDArray[np.float64], NDArray[np.float64],
]:
    """주기 경계 Helmholtz 분해.

    Returns:
        (u_sol, v_sol, u_irr, v_irr).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError("u, v 는 같은 2D 배열이어야 합니다")

    ny, nx = u.shape
    U = np.fft.fft2(u)
    V = np.fft.fft2(v)

    kx = np.fft.fftfreq(nx, d=Lx / nx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=Ly / ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0  # 평균 모드

    # irrotational part û_irr = k (k·û) / |k|²
    k_dot = KX * U + KY * V
    U_irr = KX * k_dot / K2
    V_irr = KY * k_dot / K2
    # 평균 모드는 irrotational 에 포함
    U_sol = U - U_irr
    V_sol = V - V_irr

    # 평균 처리: U[0,0] 는 그대로 u_irr 에 남김 (단순성)
    U_irr[0, 0] = U[0, 0]
    V_irr[0, 0] = V[0, 0]
    U_sol[0, 0] = 0.0
    V_sol[0, 0] = 0.0

    return (
        np.real(np.fft.ifft2(U_sol)),
        np.real(np.fft.ifft2(V_sol)),
        np.real(np.fft.ifft2(U_irr)),
        np.real(np.fft.ifft2(V_irr)),
    )


__all__ = ["helmholtz_2d"]
