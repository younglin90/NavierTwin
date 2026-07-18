"""Exact DMD — Tu et al. 2014. SVD-based rank-r approximation of Koopman.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.system_id.exact_dmd import exact_dmd
    >>> A_true = np.array([[0.9, 0.1], [-0.05, 0.85]])
    >>> x = np.array([1.0, 0.0])
    >>> X = np.column_stack((x, A_true @ x, A_true @ A_true @ x, A_true @ A_true @ A_true @ x))
    >>> res = exact_dmd(X, r=2)
    >>> res["eigenvalues"].shape
    (2,)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray


def exact_dmd(
    X: NDArray[np.float64],
    r: int | None = None,
) -> dict:
    """(n, T) 스냅샷 → DMD modes + eigenvalues."""
    X = np.asarray(X, dtype=np.float64)
    X0 = X[:, :-1]
    X1 = X[:, 1:]
    U, s, Vt = _svd(X0, full_matrices=False)
    if r is None:
        r = s.size
    r = int(min(r, s.size))
    Ur = U[:, :r]
    Sr = np.diag(s[:r])
    Vr = Vt[:r, :].T
    A_tilde = Ur.T @ X1 @ Vr @ np.linalg.inv(Sr)
    evals, W = np.linalg.eig(A_tilde)
    # exact DMD modes
    Phi = X1 @ Vr @ np.linalg.inv(Sr) @ W
    # initial amplitudes
    b = np.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]
    return {
        "eigenvalues": evals,
        "modes": Phi,
        "amplitudes": b,
        "A_tilde": A_tilde,
    }


def dmd_reconstruct(
    result: dict,
    t: NDArray[np.int64],
) -> NDArray[np.float64]:
    """t: 시간 인덱스 배열 (int). 복원: Σ_k φ_k · b_k · λ_k^t."""
    Phi = result["modes"]
    b = result["amplitudes"]
    evals = result["eigenvalues"]
    T = np.asarray(t, dtype=np.int64)
    Lam = evals[:, None] ** T[None, :]
    return np.real(Phi @ (Lam * b[:, None]))


__all__ = ["exact_dmd", "dmd_reconstruct"]
