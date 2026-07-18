"""Fourier positional encoding — PINN/NeRF 고주파 학습 개선.

[x] → [sin(2^k π x), cos(2^k π x)]_{k=0..L}.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.neural.positional_enc import fourier_encode
    >>> out = fourier_encode(np.array([[0.0, 0.5]]), num_freqs=3)
    >>> out.shape
    (1, 14)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fourier_encode(
    x: NDArray[np.float64], num_freqs: int = 6, *,
    include_input: bool = True, base: float = 2.0,
) -> NDArray[np.float64]:
    """입력 (N, d) → (N, d + 2*num_freqs*d) Fourier features."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    freqs = base ** np.arange(num_freqs) * np.pi  # (F,)
    # (N, d, F)
    xf = x[..., None] * freqs[None, None, :]
    sin = np.sin(xf).reshape(x.shape[0], -1)
    cos = np.cos(xf).reshape(x.shape[0], -1)
    parts = [sin, cos]
    if include_input:
        parts.insert(0, x)
    return np.concatenate(parts, axis=1)


def gaussian_rff(
    x: NDArray[np.float64], num_features: int = 64, *,
    sigma: float = 1.0, seed: int | None = 0,
) -> NDArray[np.float64]:
    """Gaussian Random Fourier Features — RBF kernel 근사."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0 / sigma, size=(x.shape[1], num_features))
    b = rng.uniform(0, 2 * np.pi, size=num_features)
    return np.sqrt(2.0 / num_features) * np.cos(x @ W + b)


__all__ = ["fourier_encode", "gaussian_rff"]
