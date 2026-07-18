"""Echo State Network — reservoir computing 기반 예측.

h_{t+1} = tanh(W_in x_t + W_res h_t), y_t = W_out h_t. W_out 만 학습.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.system_id.reservoir import EchoStateNetwork
    >>> rng = np.random.default_rng(0)
    >>> t = np.linspace(0, 20, 2000)
    >>> x = np.sin(t)[:, None]
    >>> y = np.cos(t)[:, None]
    >>> esn = EchoStateNetwork(n_in=1, n_res=50, n_out=1, seed=0).fit(x, y)
    >>> yhat = esn.predict(x)
    >>> np.corrcoef(yhat[:, 0], y[:, 0])[0, 1] > 0.9
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


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
    def __init__(
        self, n_in: int, n_res: int = 100, n_out: int = 1,
        *, spectral_radius: float = 0.9, sparsity: float = 0.1,
        input_scale: float = 1.0, leaky: float = 1.0, seed: int | None = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n_in = int(n_in)
        self.n_res = int(n_res)
        self.n_out = int(n_out)
        self.leaky = float(leaky)
        self.W_in = input_scale * rng.uniform(-1, 1, size=(n_res, n_in))
        W = rng.uniform(-1, 1, size=(n_res, n_res))
        mask = rng.random(size=W.shape) < sparsity
        W = W * mask
        # rescale by spectral radius
        eig = np.max(np.abs(np.linalg.eigvals(W)))
        if eig > 0:
            W = W * (spectral_radius / eig)
        self.W_res = W
        self.W_out: NDArray[np.float64] | None = None
        self.h: NDArray[np.float64] = np.zeros(n_res)

    def _run(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        T = X.shape[0]
        H = np.zeros((T, self.n_res))
        h = self.h.copy()
        t = 0
        while t < T:
            h_new = np.tanh(self.W_in @ X[t] + self.W_res @ h)
            h = (1.0 - self.leaky) * h + self.leaky * h_new
            H[t] = h
            t += 1
        self.h = h
        return H

    def fit(
        self, X: NDArray[np.float64], Y: NDArray[np.float64],
        *, ridge: float = 1e-6, discard: int = 50,
    ) -> "EchoStateNetwork":
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        H = self._run(X)
        H_ = H[discard:]
        Y_ = Y[discard:]
        # ridge regression
        self.W_out = _solve_dense_rhs(
            H_.T @ H_ + ridge * np.eye(self.n_res), H_.T @ Y_
        )
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.W_out is None:
            raise RuntimeError("fit 먼저")
        # fit 후 h 상태 이어서 예측하지 않도록 reset
        self.h = np.zeros(self.n_res)
        H = self._run(np.asarray(X, dtype=np.float64))
        return H @ self.W_out


__all__ = ["EchoStateNetwork"]
