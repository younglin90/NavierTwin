"""Complex-step 미분 — 머신 정밀도 gradient (빼기 소거 없음).

f(x + ih)/h 의 허수부가 f'(x) 를 주며, h 를 1e-30 까지 줄여도 안전.

주의: f 는 복소수 입력을 처리 가능해야 함 (대부분 NumPy 연산은 OK).

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.complex_step import cs_gradient
    >>> f = lambda x: np.sum(x ** 3)
    >>> g = cs_gradient(f, np.array([1.0, 2.0, 3.0]))
    >>> np.allclose(g, [3., 12., 27.])
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def cs_derivative(
    f: Callable[[complex], complex], x: float, h: float = 1e-30,
) -> float:
    """스칼라 f'(x)."""
    return float(np.imag(f(complex(x, h))) / h)


def cs_gradient(
    f: Callable[[NDArray], NDArray | float],
    x: NDArray[np.float64],
    h: float = 1e-30,
) -> NDArray[np.float64]:
    """벡터 f: ℝⁿ → ℝ 의 gradient."""
    x = np.asarray(x, dtype=np.float64).ravel()
    g = np.zeros_like(x)
    xc = x.astype(np.complex128)
    i = 0
    while i < x.size:
        orig = xc[i]
        xc[i] = complex(x[i], h)
        g[i] = float(np.imag(f(xc)) / h)
        xc[i] = orig
        i += 1
    return g


def cs_jacobian(
    f: Callable[[NDArray], NDArray],
    x: NDArray[np.float64],
    h: float = 1e-30,
) -> NDArray[np.float64]:
    """벡터 필드 f: ℝⁿ → ℝᵐ 의 Jacobian (m, n)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    xc = x.astype(np.complex128)
    # determine m
    f0 = np.asarray(f(xc), dtype=np.complex128).ravel()
    m = f0.size
    J = np.zeros((m, x.size))
    i = 0
    while i < x.size:
        orig = xc[i]
        xc[i] = complex(x[i], h)
        val = np.asarray(f(xc), dtype=np.complex128).ravel()
        J[:, i] = np.imag(val) / h
        xc[i] = orig
        i += 1
    return J


__all__ = ["cs_derivative", "cs_gradient", "cs_jacobian"]
