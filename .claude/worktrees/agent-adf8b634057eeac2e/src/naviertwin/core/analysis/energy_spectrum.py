"""난류 에너지 스펙트럼 — 1D/2D 파수 기반.

E(k) ≈ ½ |û(k)|² (1D) 또는 radial average (2D).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.energy_spectrum import energy_spectrum_1d
    >>> u = np.sin(2*np.pi*5*np.linspace(0,1,128,endpoint=False))
    >>> k, E = energy_spectrum_1d(u, L=1.0)
    >>> k[np.argmax(E)]  # doctest: +ELLIPSIS
    np.float64(5.0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def energy_spectrum_1d(
    u: NDArray[np.float64], L: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """1D 신호 → (k, E(k)) one-sided."""
    u = np.asarray(u, dtype=np.float64).ravel()
    n = u.size
    U = np.fft.rfft(u)
    k = np.fft.rfftfreq(n, d=L / n) * (2 * np.pi / (2 * np.pi))  # cycles/L
    E = 0.5 * np.abs(U) ** 2 / n
    return k, E


def energy_spectrum_2d_radial(
    u: NDArray[np.float64], v: NDArray[np.float64],
    Lx: float = 1.0, Ly: float = 1.0, *, n_bins: int = 64,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """2D 속도장 → 방사 평균 에너지 스펙트럼."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError("shape mismatch")
    ny, nx = u.shape
    kx = np.fft.fftfreq(nx, d=Lx / nx) * (2 * np.pi)
    ky = np.fft.fftfreq(ny, d=Ly / ny) * (2 * np.pi)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K = np.sqrt(KX ** 2 + KY ** 2)
    Uh = np.fft.fft2(u)
    Vh = np.fft.fft2(v)
    Ek = 0.5 * (np.abs(Uh) ** 2 + np.abs(Vh) ** 2) / (nx * ny) ** 2
    k_max = K.max()
    edges = np.linspace(0, k_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    E_radial = np.asarray(_kernels.radial_energy_sum(K, Ek, edges), dtype=np.float64)
    return centers, E_radial


__all__ = ["energy_spectrum_1d", "energy_spectrum_2d_radial"]
