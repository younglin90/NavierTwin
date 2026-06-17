"""NumPy 기반 경사하강법 옵티마이저 — SGD / Momentum / Adam / AdaGrad.

Framework 없이 scalar objective 최적화 (튜닝/교육용).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.gradient_opt import AdamOpt
    >>> opt = AdamOpt(lr=0.1)
    >>> x = np.array([3.0, -2.0])
    >>> x, _ = minimize(lambda z: (float(z @ z), 2 * z), x, opt, n_steps=200)
    >>> np.linalg.norm(x) < 1e-3
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SGDOpt:
    def __init__(self, lr: float = 1e-2) -> None:
        self.lr = float(lr)

    def step(self, x: NDArray, g: NDArray) -> NDArray:
        return x - self.lr * g


class MomentumOpt:
    def __init__(self, lr: float = 1e-2, momentum: float = 0.9) -> None:
        self.lr = float(lr)
        self.momentum = float(momentum)
        self._v: NDArray | None = None

    def step(self, x: NDArray, g: NDArray) -> NDArray:
        if self._v is None:
            self._v = np.zeros_like(x)
        self._v = self.momentum * self._v + g
        return x - self.lr * self._v


class AdamOpt:
    def __init__(
        self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = float(lr)
        self.b1 = float(beta1)
        self.b2 = float(beta2)
        self.eps = float(eps)
        self._m: NDArray | None = None
        self._v: NDArray | None = None
        self._t: int = 0

    def step(self, x: NDArray, g: NDArray) -> NDArray:
        if self._m is None:
            self._m = np.zeros_like(x)
            self._v = np.zeros_like(x)
        self._t += 1
        self._m = self.b1 * self._m + (1 - self.b1) * g
        self._v = self.b2 * self._v + (1 - self.b2) * (g ** 2)
        m_hat = self._m / (1 - self.b1 ** self._t)
        v_hat = self._v / (1 - self.b2 ** self._t)
        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdaGradOpt:
    def __init__(self, lr: float = 1e-2, eps: float = 1e-8) -> None:
        self.lr = float(lr)
        self.eps = float(eps)
        self._G: NDArray | None = None

    def step(self, x: NDArray, g: NDArray) -> NDArray:
        if self._G is None:
            self._G = np.zeros_like(x)
        self._G = self._G + g ** 2
        return x - self.lr * g / (np.sqrt(self._G) + self.eps)


def minimize(
    obj_grad, x0: NDArray, opt, n_steps: int = 1000, tol: float = 1e-10,
) -> tuple[NDArray, list[float]]:
    """obj_grad(x) → (f, g) 를 n_steps 회 반복."""
    x = np.asarray(x0, dtype=np.float64).copy()
    history: list[float] = []
    step = 0
    while step < n_steps:
        f, g = obj_grad(x)
        history.append(float(f))
        if np.linalg.norm(g) < tol:
            break
        x = opt.step(x, g)
        step += 1
    return x, history


__all__ = ["SGDOpt", "MomentumOpt", "AdamOpt", "AdaGradOpt", "minimize"]
