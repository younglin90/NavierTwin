"""Snapshot / dataset 결정적 train/val/test 분할.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.preprocessing.splitter import split_indices
    >>> tr, va, te = split_indices(n=100, val=0.2, test=0.1, seed=0)
    >>> len(tr), len(va), len(te)
    (70, 20, 10)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def split_indices(
    n: int,
    val: float = 0.2,
    test: float = 0.0,
    *,
    shuffle: bool = True,
    seed: int | None = 0,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """n 샘플을 (train, val, test) 인덱스로 분할."""
    if not (0.0 <= val < 1.0 and 0.0 <= test < 1.0 and val + test < 1.0):
        raise ValueError("val, test ∈ [0, 1) 그리고 val+test < 1")
    idx = np.arange(n, dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    n_val = int(round(n * val))
    n_test = int(round(n * test))
    n_train = n - n_val - n_test
    tr = idx[:n_train]
    va = idx[n_train:n_train + n_val]
    te = idx[n_train + n_val:]
    return tr, va, te


def split_snapshots(
    X: NDArray[np.float64],
    val: float = 0.2,
    test: float = 0.0,
    *,
    shuffle: bool = True,
    seed: int | None = 0,
) -> dict[str, NDArray[np.float64]]:
    """X (n_features, n_snapshots) 를 열 기준으로 분할."""
    n = X.shape[1]
    tr, va, te = split_indices(n, val, test, shuffle=shuffle, seed=seed)
    return {
        "train": X[:, tr],
        "val": X[:, va],
        "test": X[:, te],
        "train_idx": tr, "val_idx": va, "test_idx": te,
    }


def k_fold_indices(
    n: int, k: int, *, seed: int | None = 0
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """K-fold (train_idx, val_idx) 리스트."""
    if k < 2 or k > n:
        raise ValueError("2 <= k <= n")
    idx = np.arange(n, dtype=np.int64)
    np.random.default_rng(seed).shuffle(idx)
    sizes = np.full(k, n // k, dtype=np.int64)
    sizes[: n % k] += 1
    starts = np.concatenate(([0], np.cumsum(sizes[:-1])))
    stops = starts + sizes

    def _fold(bounds: tuple[np.int64, np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        start, stop = map(int, bounds)
        val = idx[start:stop]
        train = np.concatenate((idx[:start], idx[stop:]))
        return train, val

    return list(map(_fold, zip(starts, stops, strict=True)))


__all__ = ["split_indices", "split_snapshots", "k_fold_indices"]
