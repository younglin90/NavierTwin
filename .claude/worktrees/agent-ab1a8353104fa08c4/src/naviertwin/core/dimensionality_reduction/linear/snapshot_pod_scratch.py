"""Snapshot POD (Sirovich) — scratch 구현. 대형 공간 × 작은 시간 데이터에 효율적.

C = XᵀX / (N-1) 고유분해 → 모드.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.snapshot_pod_scratch import (
    ...     snapshot_pod,
    ... )
    >>> X = np.random.default_rng(0).standard_normal((100, 10))
    >>> modes, sv, coeffs = snapshot_pod(X, k=3)
    >>> modes.shape, sv.shape, coeffs.shape
    ((100, 3), (3,), (10, 3))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def snapshot_pod(
    X: NDArray[np.float64], k: int = 5,
    *, center: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """X: (n_features, n_snapshots). 반환: (modes, singular_values, coeffs)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X (n_features, n_snapshots)")
    if center:
        mean = X.mean(axis=1, keepdims=True)
        Xc = X - mean
    else:
        mean = np.zeros((X.shape[0], 1))
        Xc = X

    N = X.shape[1]
    C = (Xc.T @ Xc) / max(N - 1, 1)  # (N, N)
    evals, evecs = np.linalg.eigh(C)
    # descending
    order = np.argsort(-evals)
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]

    k = int(min(k, N))
    sv = np.sqrt(evals[:k] * max(N - 1, 1))  # scaled to match X SVD s-values
    # modes = Xc V / σ
    modes = Xc @ evecs[:, :k]
    modes = modes / (sv + 1e-30)
    coeffs = (Xc.T @ modes)  # (N, k)
    return modes, sv, coeffs


def reconstruct(
    modes: NDArray[np.float64], coeffs: NDArray[np.float64],
    mean: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """(n_features, n_snapshots) = modes @ coeffs.T + mean."""
    X = modes @ coeffs.T
    if mean is not None:
        X = X + np.asarray(mean)
    return X


__all__ = ["snapshot_pod", "reconstruct"]
