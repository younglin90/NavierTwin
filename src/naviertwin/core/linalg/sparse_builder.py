"""희소 행렬 빌더 — scipy.sparse 래퍼 + FD 스텐실 조립 헬퍼.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.sparse_builder import laplacian_1d
    >>> L = laplacian_1d(5, h=1.0)
    >>> L.shape
    (5, 5)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def laplacian_1d(n: int, h: float = 1.0, *, boundary: str = "dirichlet") -> Any:
    """1D 2차 중심차분 Laplacian (n×n)."""
    try:
        from scipy.sparse import diags
    except ImportError as exc:
        raise RuntimeError("scipy 필요") from exc
    if n < 2:
        raise ValueError("n >= 2")
    main = -2.0 * np.ones(n)
    off = 1.0 * np.ones(n - 1)
    if boundary == "periodic":
        L = diags([main, off, off], [0, -1, 1], format="lil", dtype=np.float64).tolil()
        L[0, -1] = 1.0
        L[-1, 0] = 1.0
        L = L.tocsr() / (h ** 2)
    elif boundary == "dirichlet":
        L = diags([main, off, off], [0, -1, 1], format="csr", dtype=np.float64) / (h ** 2)
    else:
        raise ValueError(f"boundary ∈ dirichlet/periodic, got {boundary}")
    return L


def laplacian_2d(
    nx: int, ny: int, hx: float = 1.0, hy: float = 1.0,
) -> Any:
    """2D 5-point stencil Laplacian — Kronecker 구조."""
    try:
        from scipy.sparse import eye, kron
    except ImportError as exc:
        raise RuntimeError("scipy 필요") from exc
    Lx = laplacian_1d(nx, hx)
    Ly = laplacian_1d(ny, hy)
    Ix = eye(nx, format="csr")
    Iy = eye(ny, format="csr")
    return kron(Iy, Lx) + kron(Ly, Ix)


def coo_to_csr(
    rows: NDArray[np.int64], cols: NDArray[np.int64],
    vals: NDArray[np.float64], shape: tuple[int, int],
) -> Any:
    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise RuntimeError("scipy 필요") from exc
    return coo_matrix((vals, (rows, cols)), shape=shape).tocsr()


__all__ = ["laplacian_1d", "laplacian_2d", "coo_to_csr"]
