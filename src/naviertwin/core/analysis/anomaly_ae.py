"""Anomaly detection — reconstruction error threshold (POD-based AE proxy).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.anomaly_ae import POD_AnomalyDetector
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 20))
    >>> det = POD_AnomalyDetector(rank=3).fit(X)
    >>> det.score(X[:, 0]).item() < 100
    True
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray


class POD_AnomalyDetector:  # noqa: N801
    def __init__(self, rank: int = 5) -> None:
        self.rank = int(rank)
        self.Phi: NDArray | None = None
        self.threshold: float = 0.0

    def fit(self, X: NDArray[np.float64]) -> "POD_AnomalyDetector":
        X = np.asarray(X, dtype=np.float64)
        U, _, _ = _svd(X, full_matrices=False)
        self.Phi = U[:, :self.rank]
        # threshold = 95th percentile of training reconstruction errors
        rec = self.Phi @ (self.Phi.T @ X)
        errs = np.linalg.norm(X - rec, axis=0)
        self.threshold = float(np.quantile(errs, 0.95))
        return self

    def score(self, x: NDArray[np.float64]) -> NDArray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return np.array(np.linalg.norm(x - self.Phi @ (self.Phi.T @ x)))
        rec = self.Phi @ (self.Phi.T @ x)
        return np.linalg.norm(x - rec, axis=0)

    def is_anomaly(self, x: NDArray) -> NDArray:
        return self.score(x) > self.threshold


__all__ = ["POD_AnomalyDetector"]
