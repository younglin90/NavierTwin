"""2D Poisson 솔버 — FFT 기반 (Dirichlet 0) + Jacobi iter.

∇²p = f. Fourier: p̂ = -f̂ / (k_x² + k_y²), k=0 mode 는 0 으로.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.pressure_poisson import poisson_2d_fft
    >>> nx, ny = 32, 32
    >>> x = np.linspace(0, 1, nx, endpoint=False)
    >>> y = np.linspace(0, 1, ny, endpoint=False)
    >>> X, Y = np.meshgrid(x, y, indexing='ij')
    >>> f = -2*(np.pi**2) * np.sin(np.pi*X) * np.sin(np.pi*Y)  # periodic context
    >>> p = poisson_2d_fft(f, Lx=1.0, Ly=1.0)
    >>> p.shape
    (32, 32)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def poisson_2d_fft(
    f: NDArray[np.float64], Lx: float = 1.0, Ly: float = 1.0,
) -> NDArray[np.float64]:
    """주기 도메인 2D Poisson ∇²p = f. f mean 이 0 이어야 해가 유일.

    Returns:
        p (mean=0).
    """
    f = np.asarray(f, dtype=np.float64)
    nx, ny = f.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0  # avoid division by zero; DC set to 0 below
    F = np.fft.fft2(f)
    P = -F / K2
    P[0, 0] = 0.0
    return np.real(np.fft.ifft2(P))


def poisson_2d_jacobi(
    f: NDArray[np.float64], dx: float = 1.0, dy: float = 1.0,
    *, max_iter: int = 5000, tol: float = 1e-6,
) -> tuple[NDArray[np.float64], dict]:
    """Dirichlet 0 경계 2D Poisson — Jacobi iteration."""
    return _kernels.poisson_2d_jacobi(
        np.asarray(f, dtype=np.float64),
        float(dx),
        float(dy),
        int(max_iter),
        float(tol),
    )


__all__ = ["poisson_2d_fft", "poisson_2d_jacobi"]
