"""Drift monitor — KS statistic + Population Stability Index (PSI).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.twin.drift_monitor import ks_stat, psi
    >>> rng = np.random.default_rng(0)
    >>> ks_stat(rng.normal(0, 1, 100), rng.normal(0, 1, 100)) >= 0
    np.True_
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ks_stat(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    a = np.sort(np.asarray(a))
    b = np.sort(np.asarray(b))
    grid = np.concatenate([a, b])
    Fa = np.searchsorted(a, grid, side="right") / len(a)
    Fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(Fa - Fb)))


def psi(
    expected: NDArray[np.float64], actual: NDArray[np.float64],
    *, n_bins: int = 10,
) -> float:
    e = np.asarray(expected)
    a = np.asarray(actual)
    edges = np.quantile(e, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    eb = np.histogram(e, bins=edges)[0] / len(e)
    ab = np.histogram(a, bins=edges)[0] / len(a)
    eb = np.maximum(eb, 1e-6)
    ab = np.maximum(ab, 1e-6)
    return float(np.sum((ab - eb) * np.log(ab / eb)))


__all__ = ["ks_stat", "psi"]
