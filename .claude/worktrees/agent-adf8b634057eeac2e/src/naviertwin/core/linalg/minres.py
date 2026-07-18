"""MINRES — symmetric indefinite 선형시스템 (Paige-Saunders 1975).

간단 버전: scipy 가 있으면 사용, 없으면 fallback Lanczos-based.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.minres import minres
    >>> A = np.array([[1., 2.], [2., 1.]])  # symmetric indefinite
    >>> b = np.array([1., 2.])
    >>> x, info = minres(A, b)
    >>> info["converged"]
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def minres(
    A: NDArray[np.float64], b: NDArray[np.float64],
    *, x0: NDArray | None = None, max_iter: int | None = None,
    tol: float = 1e-10,
) -> tuple[NDArray[np.float64], dict]:
    """scipy.sparse.linalg.minres wrapper with numpy fallback (GMRES)."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    try:
        from scipy.sparse.linalg import minres as _sp_minres

        x, flag = _sp_minres(
            A, b, x0=x0, rtol=tol, maxiter=max_iter or 5 * b.size,
        )
        res = float(np.linalg.norm(A @ x - b))
        return x, {"iters": -1, "residual": res, "converged": res < 1e-6}
    except ImportError:
        # fallback: GMRES-style (since A is symmetric, MINRES ≈ GMRES here)
        from naviertwin.core.linalg.newton_krylov import gmres
        x = gmres(lambda v: A @ v, b, max_iter=max_iter or b.size, tol=tol)
        res = float(np.linalg.norm(A @ x - b))
        return x, {"iters": -1, "residual": res, "converged": res < 1e-6}


__all__ = ["minres"]
