"""Knowledge distillation loss — KL divergence on softmax with temperature.

L = α CE(student, y) + (1-α) T² KL(softmax(s/T) || softmax(t/T)).

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.distillation import distill_loss
    >>> s = np.array([1.0, 2.0])
    >>> t = np.array([1.5, 1.5])
    >>> distill_loss(s, t, T=2.0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def distill_loss(
    student_logits: NDArray[np.float64],
    teacher_logits: NDArray[np.float64],
    *,
    T: float = 2.0,
) -> float:
    s = _softmax(np.asarray(student_logits) / T)
    t = _softmax(np.asarray(teacher_logits) / T)
    kl = np.sum(t * (np.log(t + 1e-12) - np.log(s + 1e-12)))
    return float(T * T * kl)


__all__ = ["distill_loss"]
