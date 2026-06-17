"""Linear MPC — QP 없이 최소자승으로 작은 horizon 문제 해결.

min Σ (y_k - r_k)ᵀ Q (y_k - r_k) + u_kᵀ R u_k
s.t. x_{k+1} = A x_k + B u_k, y_k = C x_k.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.control.mpc_linear import mpc_step
    >>> A = np.eye(1); B = np.array([[1.0]]); C = np.eye(1)
    >>> u = mpc_step(A, B, C, x=np.array([0.0]), ref=np.ones((5, 1)), Q=1.0, R=0.01)
    >>> u.shape
    (5, 1)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mpc_step(
    A: NDArray[np.float64], B: NDArray[np.float64], C: NDArray[np.float64],
    x: NDArray[np.float64], ref: NDArray[np.float64],
    Q: float = 1.0, R: float = 0.01,
) -> NDArray[np.float64]:
    """Horizon N = len(ref). 최소자승으로 u_0..u_{N-1} 계산."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref[:, None]
    N = ref.shape[0]
    nx = A.shape[0]
    nu = B.shape[1]
    ny = C.shape[0]

    # predict: y_k = C (A^k x + Σ A^{k-1-i} B u_i)
    # build block matrices: Ybar = Phi x + Psi U
    Phi = np.zeros((N * ny, nx))
    Psi = np.zeros((N * ny, N * nu))
    powers = [np.eye(nx)]
    while len(powers) < N:
        powers.append(A @ powers[-1])

    k = 0
    while k < N:
        Phi[k * ny:(k + 1) * ny] = C @ powers[k]
        i = 0
        while i <= k:
            Ai = powers[k - i]
            Psi[k * ny:(k + 1) * ny, i * nu:(i + 1) * nu] = C @ Ai @ B
            i += 1
        k += 1

    # min ‖Ybar - Rbar‖_Q² + ‖U‖_R²
    # LS: [sqrt(Q) Psi;  sqrt(R) I] U = [sqrt(Q) (Rbar - Phi x); 0]
    Rbar = ref.reshape(-1)
    sQ = np.sqrt(Q)
    sR = np.sqrt(R)
    Aug = np.vstack([sQ * Psi, sR * np.eye(N * nu)])
    b = np.concatenate([sQ * (Rbar - Phi @ x), np.zeros(N * nu)])
    U, *_ = np.linalg.lstsq(Aug, b, rcond=None)
    return U.reshape(N, nu)


__all__ = ["mpc_step"]
