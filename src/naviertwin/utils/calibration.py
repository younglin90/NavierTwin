"""Expected Calibration Error + temperature scaling.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.calibration import ece, temperature_scale
    >>> probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    >>> labels = np.array([0, 0])
    >>> ece(probs, labels) >= 0
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ece(
    probs: NDArray[np.float64], labels: NDArray[np.int_], *, n_bins: int = 10,
) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels)
    pred = np.argmax(p, axis=1)
    conf = np.max(p, axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(y)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            acc = correct[mask].mean()
            avg_conf = conf[mask].mean()
            e += (mask.sum() / n) * abs(acc - avg_conf)
    return float(e)


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def temperature_scale(
    logits: NDArray[np.float64], labels: NDArray[np.int_], *,
    T_grid: NDArray[np.float64] | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """Find T minimizing NLL on val set; returns (T*, calibrated probs)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    Ts = T_grid if T_grid is not None else np.linspace(0.5, 5.0, 50)
    best_T = 1.0
    best_nll = np.inf
    for T in Ts:
        p = _softmax(z / T)
        nll = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)
    return best_T, _softmax(z / best_T)


__all__ = ["ece", "temperature_scale"]
