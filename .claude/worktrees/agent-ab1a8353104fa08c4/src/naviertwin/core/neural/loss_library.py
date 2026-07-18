"""NumPy/PyTorch 호환 loss function 라이브러리.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.neural.loss_library import mse, mae, huber
    >>> mse(np.zeros(3), np.ones(3))
    1.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mse(y_true: NDArray, y_pred: NDArray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true: NDArray, y_pred: NDArray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def huber(y_true: NDArray, y_pred: NDArray, delta: float = 1.0) -> float:
    d = np.asarray(y_true) - np.asarray(y_pred)
    abs_d = np.abs(d)
    quadratic = np.minimum(abs_d, delta)
    linear = abs_d - quadratic
    return float(np.mean(0.5 * quadratic ** 2 + delta * linear))


def rmse(y_true: NDArray, y_pred: NDArray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def relative_l2(y_true: NDArray, y_pred: NDArray) -> float:
    num = np.linalg.norm(np.asarray(y_true) - np.asarray(y_pred))
    den = np.linalg.norm(np.asarray(y_true)) + 1e-30
    return float(num / den)


def quantile_loss(y_true: NDArray, y_pred: NDArray, q: float = 0.5) -> float:
    d = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def logcosh(y_true: NDArray, y_pred: NDArray) -> float:
    d = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.log(np.cosh(d))))


def smape(y_true: NDArray, y_pred: NDArray) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    denom = (np.abs(yt) + np.abs(yp)) + 1e-30
    return float(np.mean(2.0 * np.abs(yt - yp) / denom))


__all__ = ["mse", "mae", "huber", "rmse", "relative_l2", "quantile_loss",
           "logcosh", "smape"]
