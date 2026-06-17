"""DMDc — Dynamic Mode Decomposition with control.

X_{k+1} = A X_k + B U_k, lstsq [A B] = X_{k+1} [X_k; U_k]⁻¹.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.system_id.dmdc import fit_dmdc
    >>> # 구성은 test 참조
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fit_dmdc(
    X: NDArray[np.float64],
    U: NDArray[np.float64],
    *, rcond: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """X (n_state, T), U (n_ctrl, T-1) → (A, B).

    X[:, :-1] 을 X0, X[:, 1:] 를 X1, U 는 (n_ctrl, T-1).
    [A B] = X1 @ pinv([X0; U]).
    """
    X = np.asarray(X, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    if X.ndim != 2 or U.ndim != 2:
        raise ValueError("X (n_state, T), U (n_ctrl, T-1)")
    X0 = X[:, :-1]
    X1 = X[:, 1:]
    if U.shape[1] != X0.shape[1]:
        raise ValueError(f"U 길이 불일치: {U.shape[1]} vs {X0.shape[1]}")
    Omega = np.vstack([X0, U])  # (n_state + n_ctrl, T-1)
    AB, *_ = np.linalg.lstsq(Omega.T, X1.T, rcond=rcond)
    AB = AB.T  # (n_state, n_state+n_ctrl)
    n = X.shape[0]
    return AB[:, :n], AB[:, n:]


def rollout_dmdc(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    x0: NDArray[np.float64],
    U: NDArray[np.float64],
) -> NDArray[np.float64]:
    """x0, U sequence → 예측 궤적."""
    T = U.shape[1]
    out = np.zeros((x0.size, T + 1))
    out[:, 0] = x0
    k = 0
    while k < T:
        out[:, k + 1] = A @ out[:, k] + B @ U[:, k]
        k += 1
    return out


__all__ = ["fit_dmdc", "rollout_dmdc"]
