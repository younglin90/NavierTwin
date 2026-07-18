"""Ensemble / Mixture of Experts (MoE) surrogate.

단순 평균 Ensemble:
    ŷ = (1/M) Σ_m f_m(x)

MoE (soft gating):
    ŷ = Σ_m g_m(x) · f_m(x)

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.surrogate.ensemble import EnsembleSurrogate
    >>> from naviertwin.core.surrogate.rbf_surrogate import RBFSurrogate
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-1, 1, (30, 2))
    >>> y = (X[:, 0] ** 2 + X[:, 1]).reshape(-1, 1)
    >>> ens = EnsembleSurrogate([RBFSurrogate(), RBFSurrogate(), RBFSurrogate()])
    >>> ens.fit(X, y)
    >>> ens.predict(X[:3]).shape
    (3, 1)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _fit_bootstrap(args: tuple[object, NDArray[np.float64], NDArray[np.float64]]) -> None:
    model, Xb, yb = args
    model.fit(Xb, yb)


class EnsembleSurrogate:
    """균등 평균 앙상블."""

    def __init__(self, models: list[object]) -> None:
        if not models:
            raise ValueError("models 가 비었습니다")
        self.models = models
        self.is_fitted: bool = False

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        rng = np.random.default_rng(0)
        indices = rng.choice(len(X), size=(len(self.models), len(X)), replace=True)
        tuple(map(_fit_bootstrap, zip(self.models, X[indices], y[indices], strict=True)))
        self.is_fitted = True
        logger.info("EnsembleSurrogate 학습 완료: %d 모델", len(self.models))

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출")
        preds = tuple(map(lambda m: np.asarray(m.predict(X)), self.models))
        stacked = np.stack(preds, axis=0)
        return stacked.mean(axis=0)

    def predict_with_std(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출")
        preds = tuple(map(lambda m: np.asarray(m.predict(X)), self.models))
        stacked = np.stack(preds, axis=0)
        return stacked.mean(axis=0), stacked.std(axis=0)


class MixtureOfExperts:
    """각 expert 에 k-means gating 으로 입력 영역별 특화."""

    def __init__(
        self,
        experts: list[object],
        n_clusters: int | None = None,
        seed: int | None = 0,
    ) -> None:
        self.experts = experts
        self.n_clusters = n_clusters or len(experts)
        self.seed = seed
        self._centroids: NDArray[np.float64] | None = None
        self.is_fitted: bool = False

    def _assign(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        assert self._centroids is not None
        # Broadcasted squared distance matrix.
        dif = X[:, None, :] - self._centroids[None, :, :]
        d2 = np.sum(dif ** 2, axis=-1)
        return np.argmin(d2, axis=1)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(X))
        centroids = [X[idx]]
        cluster_idx = 1
        while cluster_idx < self.n_clusters:
            dists = np.min(
                np.sum((X[:, None, :] - np.array(centroids)[None, :, :]) ** 2, axis=-1),
                axis=1,
            )
            prob = dists / max(dists.sum(), 1e-30)
            next_idx = rng.choice(len(X), p=prob)
            centroids.append(X[next_idx])
            cluster_idx += 1
        self._centroids = np.array(centroids)
        step = 0
        while step < 10:
            labels = self._assign(X)
            sums = np.zeros_like(self._centroids)
            np.add.at(sums, labels, X)
            counts = np.bincount(labels, minlength=self.n_clusters)
            active = counts > 0
            self._centroids[active] = sums[active] / counts[active, None]
            step += 1

        labels = self._assign(X)
        k = 0
        while k < self.n_clusters:
            ex = self.experts[k]
            sel = labels == k
            if sel.sum() < 2:
                ex.fit(X, y)
            else:
                ex.fit(X[sel], y[sel])
            k += 1
        self.is_fitted = True
        logger.info("MoE 학습 완료: %d experts", self.n_clusters)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출")
        X = np.asarray(X, dtype=np.float64)
        labels = self._assign(X)
        preds: NDArray[np.float64] | None = None
        k = 0
        while k < self.n_clusters:
            sel = labels == k
            if sel.any():
                block = np.asarray(self.experts[k].predict(X[sel]))
                block = np.atleast_2d(block)
                if block.shape[0] != int(sel.sum()):
                    block = block.reshape(int(sel.sum()), -1)
                if preds is None:
                    preds = np.empty((X.shape[0], block.shape[1]), dtype=block.dtype)
                preds[sel] = block
            k += 1
        if preds is None:
            return np.empty((0, 0), dtype=np.float64)
        return preds


__all__ = ["EnsembleSurrogate", "MixtureOfExperts"]
