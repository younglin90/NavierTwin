"""Exact DMD — Tu et al. 2014. SVD-based rank-r approximation of Koopman.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.system_id.exact_dmd import exact_dmd
    >>> A_true = np.array([[0.9, 0.1], [-0.05, 0.85]])
    >>> x = np.array([1.0, 0.0])
    >>> traj = [x]
    >>> for _ in range(30):
    ...     traj.append(A_true @ traj[-1])
    >>> X = np.array(traj).T
    >>> res = exact_dmd(X, r=2)
    >>> res["eigenvalues"].shape
    (2,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def exact_dmd(
    X: NDArray[np.float64], r: int | None = None,
) -> dict:
    """(n, T) 스냅샷 → DMD modes + eigenvalues."""
    X = np.asarray(X, dtype=np.float64)
    X0 = X[:, :-1]
    X1 = X[:, 1:]
    U, s, Vt = np.linalg.svd(X0, full_matrices=False)
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
    result: dict, t: NDArray[np.int64],
) -> NDArray[np.float64]:
    """t: 시간 인덱스 배열 (int). 복원: Σ_k φ_k · b_k · λ_k^t."""
    Phi = result["modes"]
    b = result["amplitudes"]
    evals = result["eigenvalues"]
    T = np.asarray(t, dtype=np.int64)
    # (n_features, len(T))
    Lam = np.stack([evals ** int(tt) for tt in T], axis=1)  # (r, len(T))
    return np.real(Phi @ (Lam * b[:, None]))


__all__ = ["exact_dmd", "dmd_reconstruct"]
