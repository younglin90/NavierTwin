"""Shifted POD — Reiss et al. 2018, traveling structure decomposition.

X(x, t) ≈ Σ_k T_{c_k(t)} u_k(x).  여기서는 단일 shift 추정만 (1-frame, c(t) 추정 후 정렬).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.shifted_pod import (
    ...     estimate_shifts,
    ... )
    >>> x = np.linspace(0, 2*np.pi, 64)
    >>> t = np.arange(20)
    >>> X = np.sin(x[:, None] - 0.1 * t[None, :])
    >>> shifts = estimate_shifts(X)
    >>> shifts.shape
    (20,)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray


def estimate_shifts(X: NDArray[np.float64]) -> NDArray[np.int_]:
    """Cross-correlation argmax against first frame → integer pixel shifts."""
    X = np.asarray(X, dtype=np.float64)
    n, m = X.shape
    ref = X[:, 0]
    spectra = np.fft.fft(X, axis=0)
    ref_spectrum = np.conj(np.fft.fft(ref))[:, None]
    corr = np.real(np.fft.ifft(spectra * ref_spectrum, axis=0))
    idx = np.argmax(corr, axis=0).astype(int)
    return np.where(idx > n // 2, idx - n, idx)


def shifted_pod(
    X: NDArray[np.float64], rank: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """간단 shifted-POD: shift 추정 → 정렬 → POD."""
    shifts = estimate_shifts(X)
    n, m = X.shape
    rows = (np.arange(n)[:, None] + shifts[None, :]) % n
    X_aligned = np.take_along_axis(np.asarray(X, dtype=np.float64), rows, axis=0)
    U, _, _ = _svd(X_aligned, full_matrices=False)
    return U[:, :rank], shifts


__all__ = ["estimate_shifts", "shifted_pod"]
