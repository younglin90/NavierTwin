"""Pseudo-labeling — keep predictions above confidence + multi-aug consistency.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.pseudo_label import pseudo_label_filter
    >>> probs = np.array([[0.9, 0.1], [0.55, 0.45]])
    >>> idx, lbl = pseudo_label_filter(probs, threshold=0.7)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pseudo_label_filter(
    probs: NDArray[np.float64], *, threshold: float = 0.9,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    p = np.asarray(probs)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    keep = np.where(conf >= threshold)[0]
    return keep, pred[keep]


def consistency_filter(
    probs_a: NDArray[np.float64], probs_b: NDArray[np.float64], *,
    threshold: float = 0.9,
) -> NDArray[np.int_]:
    """Keep where both predict same class above threshold."""
    pa = np.asarray(probs_a)
    pb = np.asarray(probs_b)
    keep = (pa.max(axis=1) >= threshold) & (pb.max(axis=1) >= threshold) \
            & (pa.argmax(axis=1) == pb.argmax(axis=1))
    return np.where(keep)[0]


__all__ = ["consistency_filter", "pseudo_label_filter"]
