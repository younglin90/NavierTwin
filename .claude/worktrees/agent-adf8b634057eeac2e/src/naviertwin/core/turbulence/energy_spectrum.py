"""1D/2D 에너지 스펙트럼 E(k) 계산 + Kolmogorov -5/3 검증.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.turbulence.energy_spectrum import energy_spectrum_2d
    >>> rng = np.random.default_rng(0)
    >>> u = rng.standard_normal((64, 64))
    >>> v = rng.standard_normal((64, 64))
    >>> k, E = energy_spectrum_2d(u, v)
    >>> k.shape == E.shape
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by turbulence energy spectra")


def energy_spectrum_1d(
    u: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """1D 시그널 u(x) 의 1D 에너지 스펙트럼 E(k) = |û(k)|²."""
    u = np.asarray(u, dtype=np.float64).ravel()
    U = np.fft.rfft(u, norm="forward")
    E = np.abs(U) ** 2
    k = np.arange(E.size)
    return k.astype(np.float64), E


def energy_spectrum_2d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """2D 속도장 (u, v) 의 isotropic 스펙트럼 E(k) — radial average.

    Returns:
        (k_bins, E(k)) — k_bins 는 정수 wavenumber.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError("u, v 는 같은 2D 여야 합니다")

    ny, nx = u.shape
    U = np.fft.fft2(u) / (nx * ny)
    V = np.fft.fft2(v) / (nx * ny)
    E_k = 0.5 * (np.abs(U) ** 2 + np.abs(V) ** 2)  # (ny, nx) per-mode energy

    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    # radial binning
    k_bins = np.arange(1, min(nx, ny) // 2 + 1)
    if k_bins.size == 0:
        return k_bins.astype(np.float64), np.zeros(0, dtype=np.float64)
    edges = np.concatenate([k_bins.astype(np.float64) - 0.5, [k_bins[-1] + 0.5]])
    E = np.asarray(_kernels.radial_energy_sum(K, E_k, edges), dtype=np.float64)
    return k_bins.astype(np.float64), E


def kolmogorov_slope(
    k: NDArray[np.float64],
    E: NDArray[np.float64],
    k_min_ratio: float = 0.1,
    k_max_ratio: float = 0.5,
) -> float:
    """log-log 기울기 추정 (-5/3 근처여야 turbulent inertial subrange)."""
    k = np.asarray(k, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    mask = (k > 0) & (E > 0)
    k = k[mask]
    E = E[mask]
    if k.size < 4:
        return 0.0
    k_lo = int(k_min_ratio * k.size)
    k_hi = int(k_max_ratio * k.size)
    if k_hi - k_lo < 2:
        return 0.0
    logk = np.log(k[k_lo:k_hi])
    logE = np.log(E[k_lo:k_hi])
    slope, _ = np.polyfit(logk, logE, 1)
    return float(slope)


__all__ = ["energy_spectrum_1d", "energy_spectrum_2d", "kolmogorov_slope"]
