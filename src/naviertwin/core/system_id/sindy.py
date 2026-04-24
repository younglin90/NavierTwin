"""SINDy — sparse 회귀로 ẋ = Ξ Θ(x) 의 희소 계수 Ξ 추정.

Brunton, Proctor, Kutz 2016. STLS (sequential threshold LS) 방식.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.system_id.sindy import polynomial_library, stls
    >>> t = np.linspace(0, 10, 500)
    >>> x = np.exp(-0.5 * t)
    >>> dx = -0.5 * x
    >>> Theta = polynomial_library(x[:, None], degree=2)
    >>> Xi = stls(Theta, dx[:, None], threshold=0.05)
    >>> Xi.shape[1]
    1
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def polynomial_library(
    X: NDArray[np.float64], degree: int = 2, *, include_bias: bool = True,
) -> NDArray[np.float64]:
    """다항 라이브러리 — 단변량/다변량 모두. (n, m_features)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    cols: list[NDArray] = []
    names: list[str] = []
    if include_bias:
        cols.append(np.ones(n))
        names.append("1")
    # degree 1..degree
    for deg in range(1, degree + 1):
        # all monomials of total degree = deg (with repetition)
        from itertools import combinations_with_replacement
        for idx in combinations_with_replacement(range(d), deg):
            col = np.ones(n)
            for i in idx:
                col = col * X[:, i]
            cols.append(col)
            names.append("*".join(f"x{i}" for i in idx))
    return np.stack(cols, axis=1)


def stls(
    Theta: NDArray[np.float64], dX: NDArray[np.float64],
    *, threshold: float = 0.1, max_iter: int = 10, reg: float = 0.0,
) -> NDArray[np.float64]:
    """Sequential Thresholded Least Squares.

    Args:
        Theta: (n, m) 라이브러리.
        dX: (n, k) 목표 미분.
        threshold: |ξ_ij| < thr 인 항 zero-out.

    Returns:
        Ξ: (m, k).
    """
    Theta = np.asarray(Theta, dtype=np.float64)
    dX = np.asarray(dX, dtype=np.float64)
    if dX.ndim == 1:
        dX = dX[:, None]
    if reg > 0:
        n, m = Theta.shape
        # ridge
        Xi = np.linalg.solve(Theta.T @ Theta + reg * np.eye(m), Theta.T @ dX)
    else:
        Xi, *_ = np.linalg.lstsq(Theta, dX, rcond=None)
    for _ in range(max_iter):
        small = np.abs(Xi) < threshold
        Xi_new = Xi.copy()
        Xi_new[small] = 0.0
        # 각 target 별 활성 인덱스로 재-회귀
        for k in range(Xi.shape[1]):
            big = ~small[:, k]
            if big.sum() == 0:
                continue
            sol, *_ = np.linalg.lstsq(Theta[:, big], dX[:, k], rcond=None)
            Xi_new[big, k] = sol
        if np.allclose(Xi_new, Xi):
            break
        Xi = Xi_new
    return Xi


__all__ = ["polynomial_library", "stls"]
