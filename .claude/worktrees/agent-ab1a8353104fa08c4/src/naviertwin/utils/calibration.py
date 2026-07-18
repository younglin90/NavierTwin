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
    n = len(y)
    bin_ids = np.searchsorted(bins, conf, side="left") - 1
    valid = (bin_ids >= 0) & (bin_ids < n_bins)
    counts = np.bincount(bin_ids[valid], minlength=n_bins).astype(float)
    correct_sum = np.bincount(bin_ids[valid], weights=correct[valid], minlength=n_bins)
    conf_sum = np.bincount(bin_ids[valid], weights=conf[valid], minlength=n_bins)
    nonzero = counts > 0
    acc = np.divide(correct_sum, counts, out=np.zeros_like(correct_sum), where=nonzero)
    avg_conf = np.divide(conf_sum, counts, out=np.zeros_like(conf_sum), where=nonzero)
    e = np.sum((counts / max(n, 1)) * np.abs(acc - avg_conf))
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
    Ts = np.asarray(T_grid if T_grid is not None else np.linspace(0.5, 5.0, 50))
    probs_grid = _softmax(z[None, :, :] / Ts[:, None, None], axis=2)
    nll = -np.mean(
        np.log(
            probs_grid[
                np.arange(len(Ts))[:, None],
                np.arange(len(y))[None, :],
                y[None, :],
            ]
            + 1e-12
        ),
        axis=1,
    )
    best_T = float(Ts[int(np.argmin(nll))])
    return best_T, _softmax(z / best_T)


__all__ = ["ece", "temperature_scale"]
