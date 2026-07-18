"""Spectral Galerkin — 주기 경계, Fourier basis.

du/dt = ν u_xx + f(u),  x ∈ [0, 2π).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.spectral_galerkin import (
    ...     fourier_diff, heat_step_fourier,
    ... )
    >>> x = np.linspace(0, 2*np.pi, 32, endpoint=False)
    >>> u = np.sin(x)
    >>> ux = fourier_diff(u, order=1, L=2*np.pi)
    >>> np.allclose(ux, np.cos(x), atol=1e-10)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fourier_diff(
    u: NDArray[np.float64], *, order: int = 1, L: float = 2 * np.pi,
) -> NDArray[np.float64]:
    """주기 도함수 (Fourier spectral)."""
    u = np.asarray(u, dtype=np.float64)
    n = u.shape[0]
    k = np.fft.fftfreq(n, d=L / n) * 2 * np.pi
    uk = np.fft.fft(u)
    duk = (1j * k) ** order * uk
    return np.real(np.fft.ifft(duk))


def heat_step_fourier(
    u: NDArray[np.float64], *, dt: float, nu: float = 1.0, L: float = 2 * np.pi,
) -> NDArray[np.float64]:
    """주기 열방정식 정확 한 스텝: û(t+dt) = exp(-ν k² dt) û(t)."""
    u = np.asarray(u, dtype=np.float64)
    n = u.shape[0]
    k = np.fft.fftfreq(n, d=L / n) * 2 * np.pi
    uk = np.fft.fft(u)
    uk_new = np.exp(-nu * k * k * dt) * uk
    return np.real(np.fft.ifft(uk_new))


__all__ = ["fourier_diff", "heat_step_fourier"]
