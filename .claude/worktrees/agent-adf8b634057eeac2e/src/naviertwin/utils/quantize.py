"""INT8 quantization — symmetric per-tensor.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.quantize import quantize_int8, dequantize_int8
    >>> x = np.array([-1.0, 0.0, 1.0])
    >>> q, s = quantize_int8(x)
    >>> dequantize_int8(q, s)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def quantize_int8(x: NDArray[np.float64]) -> tuple[NDArray[np.int8], float]:
    x = np.asarray(x, dtype=np.float64)
    scale = float(np.abs(x).max() / 127.0) if x.size > 0 and np.abs(x).max() > 0 else 1.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q: NDArray[np.int8], scale: float) -> NDArray[np.float64]:
    return q.astype(np.float64) * scale


__all__ = ["dequantize_int8", "quantize_int8"]
