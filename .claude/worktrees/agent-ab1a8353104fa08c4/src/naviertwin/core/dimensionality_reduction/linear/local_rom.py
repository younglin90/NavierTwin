"""Local (clustered) ROM — k-means + per-cluster POD basis.

각 snapshot 을 클러스터에 할당 후 클러스터별 POD 베이스 학습.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.local_rom import LocalROM
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 30))
    >>> lr = LocalROM(n_clusters=2, rank=3).fit(X)
    >>> lr.encode(X[:, 0]).shape
    (3,)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray


def _kmeans(
    X: NDArray[np.float64], k: int, *, n_iter: int = 30, seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Lloyd's algorithm on column vectors of X."""
    rng = np.random.default_rng(seed)
    m = X.shape[1]
    centers = X[:, rng.choice(m, size=k, replace=False)].copy()
    labels = np.zeros(m, dtype=int)
    iteration = 0
    while iteration < n_iter:
        d = np.linalg.norm(X[:, :, None] - centers[:, None, :], axis=0)  # (m, k)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        counts = np.bincount(labels, minlength=k)
        membership = (labels[:, None] == np.arange(k)).astype(np.float64)
        nonempty = counts > 0
        centers[:, nonempty] = (X @ membership[:, nonempty]) / counts[nonempty]
        iteration += 1
    return centers, labels


class LocalROM:
    def __init__(self, n_clusters: int = 3, rank: int = 5, seed: int = 0) -> None:
        self.n_clusters = int(n_clusters)
        self.rank = int(rank)
        self.seed = int(seed)
        self.centers: NDArray | None = None
        self.bases: list[NDArray] = []

    def fit(self, X: NDArray[np.float64]) -> "LocalROM":
        X = np.asarray(X, dtype=np.float64)
        self.centers, labels = _kmeans(X, self.n_clusters, seed=self.seed)
        self.bases = []
        j = 0
        while j < self.n_clusters:
            Xj = X[:, labels == j]
            if Xj.shape[1] == 0:
                self.bases.append(np.zeros((X.shape[0], self.rank)))
                j += 1
                continue
            U, _, _ = _svd(Xj, full_matrices=False)
            r = min(self.rank, U.shape[1])
            B = np.zeros((X.shape[0], self.rank))
            B[:, :r] = U[:, :r]
            self.bases.append(B)
            j += 1
        return self

    def _assign(self, x: NDArray) -> int:
        d = np.linalg.norm(self.centers - x[:, None], axis=0)
        return int(np.argmin(d))

    def encode(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=np.float64).ravel()
        j = self._assign(x)
        return self.bases[j].T @ x

    def decode(self, z: NDArray[np.float64], cluster: int) -> NDArray[np.float64]:
        return self.bases[cluster] @ np.asarray(z, dtype=np.float64)


__all__ = ["LocalROM"]
