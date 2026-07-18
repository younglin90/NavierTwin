"""선형 시스템 적합 — X_{k+1} ≈ A X_k 에서 A 추정 (least-squares DMD-like).

고차 Koopman 대신 간단한 EDMD-lite / linear predictor 로 사용.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.koopman.linear_fit import fit_linear_dynamics
    >>> x0 = np.array([1.0, 0.0])
    >>> A_true = np.array([[0.9, 0.1], [-0.1, 0.9]])
    >>> traj = [x0]
    >>> step = 0
    >>> while step < 50:
    ...     traj.append(A_true @ traj[-1])
    ...     step += 1
    >>> X = np.array(traj).T
    >>> A_hat = fit_linear_dynamics(X)
    >>> np.allclose(A_hat, A_true, atol=1e-8)
    True
"""

from __future__ import annotations

from itertools import accumulate, repeat

import numpy as np
from numpy.typing import NDArray


def fit_linear_dynamics(
    trajectory: NDArray[np.float64],
    *,
    rcond: float | None = None,
) -> NDArray[np.float64]:
    """(n_states, n_times) trajectory → A ∈ ℝ^{n×n} 최소자승."""
    X = np.asarray(trajectory, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("trajectory (n_states, n_times>=2)")
    X0 = X[:, :-1]
    X1 = X[:, 1:]
    A, *_ = np.linalg.lstsq(X0.T, X1.T, rcond=rcond)
    return A.T


def rollout_linear(
    A: NDArray[np.float64], x0: NDArray[np.float64], n_steps: int,
) -> NDArray[np.float64]:
    """x_{k+1} = A x_k 로 전개 → (n_states, n_steps+1)."""
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    states = accumulate(repeat(A, n_steps), lambda state, matrix: matrix @ state, initial=x0)
    return np.column_stack(tuple(states))


def eigenanalysis(A: NDArray[np.float64]) -> dict[str, NDArray]:
    """spectrum — eigenvalues / |λ| / 안정성 (|λ|<1)."""
    evals, evecs = np.linalg.eig(A)
    return {
        "eigenvalues": evals,
        "magnitudes": np.abs(evals),
        "eigenvectors": evecs,
        "stable": bool(np.all(np.abs(evals) < 1.0 + 1e-9)),
    }


__all__ = ["fit_linear_dynamics", "rollout_linear", "eigenanalysis"]
