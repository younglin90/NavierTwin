"""Gradient check + 수치/해석 Jacobian 비교 유틸.

PINN / 사용자 model 의 gradient 를 finite-difference 와 비교해 검증한다.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.gradient_check import finite_difference_gradient
    >>> f = lambda x: float(x[0] ** 2 + 2 * x[1])
    >>> g = finite_difference_gradient(f, np.array([1.0, 1.0]))
    >>> np.allclose(g, [2.0, 2.0], atol=1e-5)
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def finite_difference_gradient(
    f: Callable[[NDArray[np.float64]], float],
    x: NDArray[np.float64],
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    """중앙 유한차분으로 gradient 근사.

    Args:
        f: 스칼라 함수.
        x: 평가점 (n,).
        eps: perturbation.

    Returns:
        gradient 벡터 (n,).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    g = np.zeros_like(x)
    i = 0
    while i < x.size:
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (float(f(xp)) - float(f(xm))) / (2 * eps)
        i += 1
    return g


def finite_difference_jacobian(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    """중앙 차분 Jacobian (m, n)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    f0 = np.asarray(f(x), dtype=np.float64).ravel()
    m = f0.size
    n = x.size
    J = np.zeros((m, n))
    j = 0
    while j < n:
        xp = x.copy()
        xm = x.copy()
        xp[j] += eps
        xm[j] -= eps
        J[:, j] = (np.asarray(f(xp)).ravel() - np.asarray(f(xm)).ravel()) / (2 * eps)
        j += 1
    return J


def gradient_check(
    f: Callable[[NDArray[np.float64]], float],
    analytic_grad: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    eps: float = 1e-6,
    tol: float = 1e-4,
) -> dict[str, float]:
    """analytic gradient vs FD 차이 평가.

    Returns:
        {"rel_error", "max_abs_error", "ok": bool}.
    """
    fd = finite_difference_gradient(f, x, eps)
    ga = np.asarray(analytic_grad(x), dtype=np.float64).ravel()
    if ga.size != fd.size:
        raise ValueError(f"gradient size 불일치: {ga.size} vs {fd.size}")
    diff = ga - fd
    denom = np.linalg.norm(ga) + np.linalg.norm(fd) + 1e-30
    rel_err = float(np.linalg.norm(diff) / denom)
    max_abs = float(np.max(np.abs(diff)))
    ok = rel_err < tol
    logger.info(
        "gradient_check: rel_err=%.3g, max_abs=%.3g, ok=%s", rel_err, max_abs, ok,
    )
    return {"rel_error": rel_err, "max_abs_error": max_abs, "ok": float(ok)}


def torch_autograd_gradient(
    f: Callable[..., "object"],
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """PyTorch autograd 로 gradient 를 계산. f(torch.Tensor) → scalar Tensor."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc

    xt = torch.tensor(np.asarray(x, dtype=np.float64), requires_grad=True)
    val = f(xt)
    if not isinstance(val, torch.Tensor):
        raise ValueError("f 는 tensor 를 반환해야 합니다")
    if val.numel() != 1:
        val = val.sum()
    val.backward()
    return xt.grad.detach().numpy() if xt.grad is not None else np.zeros_like(x)


__all__ = [
    "finite_difference_gradient",
    "finite_difference_jacobian",
    "gradient_check",
    "torch_autograd_gradient",
]
