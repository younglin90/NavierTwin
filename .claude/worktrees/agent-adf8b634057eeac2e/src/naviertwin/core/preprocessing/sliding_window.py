"""스냅샷 시퀀스에 대한 sliding-window — 시계열 ROM/DMD/Transformer 전처리.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.preprocessing.sliding_window import make_windows
    >>> X = np.arange(20).reshape(1, 20)
    >>> W = make_windows(X, window=4, stride=2)
    >>> W.shape
    (1, 9, 4)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_windows(
    X: NDArray[np.float64],
    window: int,
    stride: int = 1,
) -> NDArray[np.float64]:
    """(n_features, T) → (n_features, n_windows, window).

    Args:
        X: 2D (features, time).
        window: 윈도우 길이.
        stride: 슬라이드 간격.
    """
    if window < 1 or stride < 1:
        raise ValueError("window/stride >= 1")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2D (features, time)")
    T = X.shape[1]
    if window > T:
        raise ValueError(f"window({window}) > T({T})")
    n_windows = (T - window) // stride + 1
    idx = np.arange(n_windows)[:, None] * stride + np.arange(window)[None, :]
    return X[:, idx]  # (features, n_windows, window)


def make_io_pairs(
    X: NDArray[np.float64],
    in_len: int,
    out_len: int,
    stride: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(input_seq, target_seq) 페어 생성 — autoregressive 학습용.

    Returns:
        inputs: (features, n_pairs, in_len),
        targets: (features, n_pairs, out_len).
    """
    total = in_len + out_len
    W = make_windows(X, window=total, stride=stride)
    return W[:, :, :in_len], W[:, :, in_len:]


__all__ = ["make_windows", "make_io_pairs"]
