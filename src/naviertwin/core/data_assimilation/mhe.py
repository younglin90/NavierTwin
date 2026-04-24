"""Moving Horizon Estimation — sliding-window Bayesian state estimate.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_assimilation.mhe import mhe_estimate
    >>> A = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> Y = np.array([[1.0], [2.0], [3.0]])
    >>> x0 = np.array([0.0, 0.0])
    >>> x = mhe_estimate(A, H, Y, x0, P0=np.eye(2), Q=0.01*np.eye(2), R=0.1*np.eye(1))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mhe_estimate(
    A: NDArray[np.float64],
    H: NDArray[np.float64],
    Y: NDArray[np.float64],
    x0: NDArray[np.float64],
    *,
    P0: NDArray[np.float64],
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Linear MHE: minimize prior + sum (z - H x_k)ᵀ R⁻¹ ... 갈, Newton."""
    N = Y.shape[0]
    Ri = np.linalg.inv(R)
    Qi = np.linalg.inv(Q)
    P0i = np.linalg.inv(P0)
    # build huge linear system in x_0..x_{N-1}; compactly via dynamic programming forward
    # information form (filter):
    Lambda = P0i.copy()
    eta = P0i @ x0
    x_curr = x0.copy()
    for k in range(N):
        # prior: x_k | x_{k-1} = A x_{k-1}, error covariance Q
        # measurement update on x_k
        z = Y[k]
        # process step: prior on x_k: mean=A x_{k-1}, cov = A P_{k-1} Aᵀ + Q
        Lambda = Qi + A.T @ Lambda @ A  # info-form approx
        eta = A.T @ eta
        # measurement
        Lambda = Lambda + H.T @ Ri @ H
        eta = eta + H.T @ Ri @ z
        x_curr = np.linalg.solve(Lambda, eta)
    return x_curr


__all__ = ["mhe_estimate"]
