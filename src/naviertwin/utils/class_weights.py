"""Class-imbalance reweighting — inverse frequency or balanced.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.class_weights import balanced_weights
    >>> balanced_weights(np.array([0, 0, 0, 1]))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def balanced_weights(y: NDArray) -> dict[int, float]:
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    n_classes = len(classes)
    weights = n / (n_classes * counts)
    return dict(zip(map(int, classes), map(float, weights), strict=True))


def per_sample_weights(y: NDArray) -> NDArray[np.float64]:
    y = np.asarray(y)
    if y.size == 0:
        return np.array([], dtype=np.float64)
    classes, inverse, counts = np.unique(y, return_counts=True, return_inverse=True)
    weights = len(y) / (len(classes) * counts)
    return np.asarray(weights[inverse], dtype=np.float64)


__all__ = ["balanced_weights", "per_sample_weights"]
