"""FastICA — 통계적 독립 성분 분리 (sklearn 선호 + numpy 폴백).

    X = A S  (A: mixing, S: 통계적으로 독립)

FastICA 로 unmixing matrix W 를 추정 → S = W X.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.ica import FastICA
    >>> rng = np.random.default_rng(0)
    >>> t = np.linspace(0, 8, 500)
    >>> s1 = np.sin(2 * t); s2 = np.sign(np.sin(3 * t))
    >>> S = np.column_stack([s1, s2])
    >>> A = np.array([[1, 0.5], [0.5, 1]])
    >>> X = S @ A.T  # mixing
    >>> ica = FastICA(n_components=2, seed=0)
    >>> S_rec = ica.fit_transform(X)
    >>> S_rec.shape
    (500, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class FastICA:
    """FastICA 단순 구현 — sklearn 있으면 그걸로."""

    def __init__(
        self,
        n_components: int,
        max_iter: int = 200,
        tol: float = 1e-5,
        seed: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        self.W_: NDArray[np.float64] | None = None
        self.whitening_: NDArray[np.float64] | None = None
        self.mean_: NDArray[np.float64] | None = None
        self.is_fitted: bool = False

    def _whiten(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # PCA whitening
        U, s, Vt = _svd(Xc, full_matrices=False)
        k = self.n_components
        self.whitening_ = (Vt[:k].T / (s[:k] + 1e-12)) * np.sqrt(Xc.shape[0])
        return Xc @ self.whitening_

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X (N, d) 2D 필요")

        # sklearn fast path
        try:
            from sklearn.decomposition import FastICA as SkFastICA

            ica = SkFastICA(
                n_components=self.n_components,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.seed,
            )
            S = ica.fit_transform(X)
            self.W_ = np.asarray(ica.components_, dtype=np.float64)
            self.is_fitted = True
            return S
        except ImportError:
            pass

        rng = np.random.default_rng(self.seed)
        Z = self._whiten(X)
        n = self.n_components
        W = rng.standard_normal((n, n))

        def g(u: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            # logcosh nonlinearity
            gx = np.tanh(u)
            dgx = 1.0 - gx ** 2
            return gx, dgx

        iteration = 0
        while iteration < self.max_iter:
            wz = Z @ W.T
            gw, dgw = g(wz)
            W_new = (gw.T @ Z) / Z.shape[0] - dgw.mean(axis=0)[:, None] * W

            # Symmetric decorrelation
            U, s, Vt = _svd(W_new)
            W_new = U @ Vt

            if np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1.0)) < self.tol:
                W = W_new
                break
            W = W_new
            iteration += 1

        self.W_ = W @ self.whitening_.T  # original-space unmixing
        self.is_fitted = True
        S = Z @ W.T
        return S


__all__ = ["FastICA"]
