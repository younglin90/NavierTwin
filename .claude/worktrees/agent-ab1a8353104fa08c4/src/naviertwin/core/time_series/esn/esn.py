"""Echo State Network (ESN) — Reservoir Computing.

무작위 희소 가중치의 large reservoir 를 고정, readout 만 학습.
빠른 학습 (ridge regression) + 긴 시계열에 효과적.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.time_series.esn.esn import EchoStateNetwork
    >>> rng = np.random.default_rng(0)
    >>> t = np.linspace(0, 50, 1000)
    >>> seq = np.column_stack([np.sin(t), np.cos(t)])
    >>> esn = EchoStateNetwork(n_features=2, reservoir_size=100, seed=0)
    >>> esn.fit(seq[:700], seq[1:701])
    >>> y = esn.predict(seq[700:-1])
    >>> y.shape == (299, 2)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _solve_dense_rhs(A: NDArray[np.float64], B: NDArray[np.float64]) -> NDArray[np.float64]:
    if _kernels is None:
        raise ImportError("naviertwin._native._kernels is required by EchoStateNetwork")
    rhs = np.asarray(B, dtype=np.float64)
    if rhs.ndim == 1:
        return _kernels.solve_dense(A, rhs)
    out = np.empty_like(rhs)
    j = 0
    while j < rhs.shape[1]:
        out[:, j] = _kernels.solve_dense(A, rhs[:, j])
        j += 1
    return out


class EchoStateNetwork:
    """Classic ESN with tanh reservoir."""

    def __init__(
        self,
        n_features: int,
        reservoir_size: int = 200,
        spectral_radius: float = 0.9,
        input_scale: float = 1.0,
        sparsity: float = 0.1,
        leak_rate: float = 1.0,
        ridge: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        self.n_features = n_features
        self.res_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.ridge = ridge
        self.rng = np.random.default_rng(seed)

        self._W_in: NDArray[np.float64] | None = None
        self._W: NDArray[np.float64] | None = None
        self._W_out: NDArray[np.float64] | None = None
        self._state: NDArray[np.float64] | None = None
        self.is_fitted: bool = False

    def _init_reservoir(self) -> None:
        N = self.res_size
        d = self.n_features
        self._W_in = self.input_scale * (self.rng.random((N, d)) * 2 - 1)
        W = self.rng.random((N, N)) * 2 - 1
        # sparsity
        W[self.rng.random((N, N)) > self.sparsity] = 0.0
        # spectral radius scaling
        eigs = np.linalg.eigvals(W)
        max_abs = float(np.max(np.abs(eigs))) if eigs.size else 1.0
        if max_abs > 0:
            W = W * (self.spectral_radius / max_abs)
        self._W = W
        self._state = np.zeros(N)

    def _update(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._state is not None and self._W is not None and self._W_in is not None
        pre = self._W_in @ u + self._W @ self._state
        self._state = (1 - self.leak_rate) * self._state + self.leak_rate * np.tanh(pre)
        return self._state

    def fit(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        warmup: int = 50,
    ) -> None:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.shape[0] != Y.shape[0] or X.shape[1] != self.n_features:
            raise ValueError("X, Y shape 또는 n_features 불일치")

        self._init_reservoir()
        states = np.zeros((X.shape[0] - warmup, self.res_size))
        t = 0
        while t < X.shape[0]:
            self._update(X[t])
            if t >= warmup:
                states[t - warmup] = self._state
            t += 1

        Yt = Y[warmup:]
        A = states.T @ states + self.ridge * np.eye(self.res_size)
        B = states.T @ Yt
        self._W_out = _solve_dense_rhs(A, B)  # (res, n_features)
        self.is_fitted = True
        logger.info(
            "ESN fit: reservoir=%d, warmup=%d, T=%d",
            self.res_size, warmup, X.shape[0],
        )

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted or self._W_out is None:
            raise RuntimeError("fit() 먼저 호출")
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros((X.shape[0], self.n_features))
        t = 0
        while t < X.shape[0]:
            s = self._update(X[t])
            out[t] = s @ self._W_out
            t += 1
        return out


__all__ = ["EchoStateNetwork"]
