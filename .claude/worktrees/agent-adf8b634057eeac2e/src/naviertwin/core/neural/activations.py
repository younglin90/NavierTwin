"""NumPy activations — ReLU / sigmoid / tanh / GELU / SiLU / ELU / softplus.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.neural.activations import relu, sigmoid, gelu
    >>> relu(np.array([-1, 2, -3, 4])).tolist()
    [0, 2, 0, 4]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def relu(x: NDArray) -> NDArray:
    return np.maximum(np.asarray(x), 0)


def leaky_relu(x: NDArray, alpha: float = 0.01) -> NDArray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 0, x, alpha * x)


def sigmoid(x: NDArray) -> NDArray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x: NDArray) -> NDArray:
    return np.tanh(np.asarray(x, dtype=np.float64))


def gelu(x: NDArray) -> NDArray:
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def silu(x: NDArray) -> NDArray:
    return np.asarray(x, dtype=np.float64) * sigmoid(x)


def elu(x: NDArray, alpha: float = 1.0) -> NDArray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def softplus(x: NDArray) -> NDArray:
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def softmax(x: NDArray, axis: int = -1) -> NDArray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


__all__ = ["relu", "leaky_relu", "sigmoid", "tanh", "gelu",
           "silu", "elu", "softplus", "softmax"]
