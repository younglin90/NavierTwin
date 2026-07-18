"""Newton-Krylov — 대형 비선형 시스템용 (GMRES + FD Jv)."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def _jv(F: Callable, x: NDArray, v: NDArray, eps: float = 1e-7) -> NDArray:
    return (F(x + eps * v) - F(x)) / eps


def gmres(
    A_fn: Callable[[NDArray], NDArray],
    b: NDArray,
    *,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    n = b.size
    max_iter = int(min(max_iter, n))
    x = np.zeros(n)
    r = b - A_fn(x)
    beta = float(np.linalg.norm(r))
    if beta < tol:
        return x
    V = [r / beta]
    H = np.zeros((max_iter + 1, max_iter))
    k_last = 0
    k = 0
    while k < max_iter:
        k_last = k + 1
        w = A_fn(V[k])
        i = 0
        while i < k + 1:
            H[i, k] = V[i] @ w
            w = w - H[i, k] * V[i]
            i += 1
        H[k + 1, k] = float(np.linalg.norm(w))
        if H[k + 1, k] < 1e-14:
            break
        V.append(w / H[k + 1, k])
        e1 = np.zeros(k + 2)
        e1[0] = beta
        y, *_ = np.linalg.lstsq(H[:k + 2, :k + 1], e1, rcond=None)
        res = float(np.linalg.norm(H[:k + 2, :k + 1] @ y - e1))
        if res < tol:
            V_mat = np.stack(V[:k + 1], axis=1)
            return x + V_mat @ y
        k += 1
    e1 = np.zeros(k_last + 1)
    e1[0] = beta
    y, *_ = np.linalg.lstsq(H[:k_last + 1, :k_last], e1, rcond=None)
    V_mat = np.stack(V[:k_last], axis=1)
    return x + V_mat @ y


def newton_krylov(
    F: Callable[[NDArray], NDArray],
    x0: NDArray,
    *,
    max_iter: int = 30,
    tol: float = 1e-8,
    gmres_iter: int = 30,
) -> tuple[NDArray, dict]:
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    k = 0
    while k < max_iter:
        fx = F(x)
        if np.linalg.norm(fx) < tol:
            return x, {"iters": k, "residual": float(np.linalg.norm(fx)), "converged": True}
        dx = gmres(
            lambda v: _jv(F, x, v),
            -fx,
            max_iter=min(gmres_iter, x.size),
            tol=tol * 0.1,
        )
        x = x + dx
        k += 1
    return x, {"iters": max_iter, "residual": float(np.linalg.norm(F(x))), "converged": False}


__all__ = ["gmres", "newton_krylov"]
