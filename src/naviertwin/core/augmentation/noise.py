"""스냅샷 데이터 증강 — 가우시안/균일/드롭아웃 노이즈.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.augmentation.noise import add_gaussian_noise
    >>> X = np.zeros((10, 5))
    >>> Y = add_gaussian_noise(X, sigma=0.1, seed=0)
    >>> Y.shape
    (10, 5)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def add_gaussian_noise(
    X: NDArray[np.float64],
    sigma: float = 0.01,
    *,
    relative: bool = False,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """X + N(0, sigma²). relative=True 면 sigma 를 X 의 std 에 곱함."""
    rng = np.random.default_rng(seed)
    s = sigma * float(np.std(X)) if relative else sigma
    return X + rng.normal(0.0, s, size=X.shape)


def add_uniform_noise(
    X: NDArray[np.float64],
    amplitude: float = 0.01,
    *,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """X + U(-amp, +amp)."""
    rng = np.random.default_rng(seed)
    return X + rng.uniform(-amplitude, amplitude, size=X.shape)


def random_dropout(
    X: NDArray[np.float64],
    drop_rate: float = 0.1,
    *,
    fill: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """random 위치를 fill 로 치환 (dropout-style)."""
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError("drop_rate ∈ [0, 1)")
    rng = np.random.default_rng(seed)
    mask = rng.random(X.shape) < drop_rate
    out = X.copy()
    out[mask] = fill
    return out


def augment_batch(
    X: NDArray[np.float64],
    n_copies: int,
    *,
    sigma: float = 0.01,
    seed: int | None = 0,
) -> NDArray[np.float64]:
    """X 에 Gaussian 노이즈 버전 n_copies 개 붙여 [X, X+n1, X+n2, ...] 반환."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=(n_copies,) + X.shape)
    noisy = X[np.newaxis, ...] + noise
    return np.concatenate((X,) + tuple(noisy), axis=-1)


__all__ = [
    "add_gaussian_noise",
    "add_uniform_noise",
    "random_dropout",
    "augment_batch",
]
