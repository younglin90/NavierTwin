"""Smoothed Particle Hydrodynamics (SPH) — 커널 기반 연속체 근사.

1D SPH 입자에서 밀도/압력 보간. 1-, 2-, 3D cubic spline kernel 제공.

    W(r, h) = (σ_d / h^d) · q(r/h)
    q(s) = {
        1 - 1.5 s² + 0.75 s³,   s ∈ [0, 1]
        0.25 (2-s)³,            s ∈ [1, 2]
        0,                       else
    }

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solver_interfaces.sph import (
    ...     cubic_spline_kernel, sph_density_1d,
    ... )
    >>> w = cubic_spline_kernel(np.array([0.0]), h=1.0, dim=1)
    >>> float(w[0])  # σ_1/h · 1 = 2/3
    0.6666666666666666
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _sigma(dim: int) -> float:
    if dim == 1:
        return 2.0 / 3.0
    if dim == 2:
        return 10.0 / (7.0 * np.pi)
    if dim == 3:
        return 1.0 / np.pi
    raise ValueError(f"dim 은 1/2/3: {dim}")


def cubic_spline_kernel(
    r: NDArray[np.float64],
    h: float,
    dim: int = 1,
) -> NDArray[np.float64]:
    """M4 cubic spline W(r, h)."""
    r = np.asarray(r, dtype=np.float64)
    s = np.abs(r) / h
    sig = _sigma(dim) / h ** dim
    out = np.zeros_like(s)
    m1 = s < 1.0
    m2 = (s >= 1.0) & (s < 2.0)
    out[m1] = sig * (1.0 - 1.5 * s[m1] ** 2 + 0.75 * s[m1] ** 3)
    out[m2] = sig * 0.25 * (2.0 - s[m2]) ** 3
    return out


def sph_density_1d(
    particles_x: NDArray[np.float64],
    masses: NDArray[np.float64],
    h: float,
) -> NDArray[np.float64]:
    """각 입자의 SPH 밀도 ρ_i = Σ_j m_j W(|x_i - x_j|, h).

    Vectorized: (N, N) 거리 행렬에 단일 kernel 호출 → O(N²) 메모리, 단일 GPU-친화.
    """
    x = np.asarray(particles_x, dtype=np.float64).ravel()
    m = np.asarray(masses, dtype=np.float64).ravel()
    if x.size != m.size:
        raise ValueError("particles, masses 크기 불일치")
    # dr[i, j] = x[j] - x[i]
    dr = x[None, :] - x[:, None]
    w = cubic_spline_kernel(dr, h, dim=1)  # (N, N)
    # ρ_i = Σ_j m_j W(...)
    return (w * m[None, :]).sum(axis=1)


def sph_gradient_1d(
    particles_x: NDArray[np.float64],
    values: NDArray[np.float64],
    masses: NDArray[np.float64],
    h: float,
) -> NDArray[np.float64]:
    """SPH 경사 근사 ∇v_i ≈ Σ_j m_j (v_j - v_i) ∇W.

    Vectorized: (N, N) 거리/속도차 행렬 broadcasting → O(N²) 메모리, 루프 제거.
    """
    x = np.asarray(particles_x, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    m = np.asarray(masses, dtype=np.float64).ravel()

    # dr[i, j] = x[j] - x[i]
    dr = x[None, :] - x[:, None]
    s = np.abs(dr) / h
    sig = _sigma(1) / h ** 2

    dw = np.zeros_like(dr)
    m1 = (s < 1.0) & (s > 0)
    m2 = (s >= 1.0) & (s < 2.0)
    dw[m1] = sig * (-3.0 * s[m1] + 2.25 * s[m1] ** 2) * np.sign(dr[m1])
    dw[m2] = -sig * 0.75 * (2.0 - s[m2]) ** 2 * np.sign(dr[m2])

    # v_diff[i, j] = v[j] - v[i]
    v_diff = v[None, :] - v[:, None]
    return (m[None, :] * v_diff * dw).sum(axis=1)


__all__ = ["cubic_spline_kernel", "sph_density_1d", "sph_gradient_1d"]
