"""Incremental SVD — Brand 2002 style (실시간 ROM 업데이트).

간단 버전: 새 열이 올 때마다 thin SVD 재계산 (rank 유지).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.incremental_svd import (
    ...     IncrementalSVD,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> isvd = IncrementalSVD(rank=3)
    >>> step = 0
    >>> while step < 10:
    ...     isvd.update(rng.standard_normal(50))
    ...     step += 1
    >>> isvd.U.shape
    (50, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray


class IncrementalSVD:
    def __init__(self, rank: int = 10) -> None:
        self.rank = int(rank)
        self.U: NDArray | None = None
        self.s: NDArray | None = None
        self.Vt: NDArray | None = None

    def update(self, col: NDArray[np.float64]) -> None:
        col = np.asarray(col, dtype=np.float64).ravel()[:, None]
        if self.U is None:
            # 초기화
            self.U = col / (np.linalg.norm(col) + 1e-30)
            self.s = np.array([float(np.linalg.norm(col))])
            self.Vt = np.array([[1.0]])
            return
        # project
        m = self.U.T @ col  # (k, 1)
        p = col - self.U @ m
        p_norm = float(np.linalg.norm(p))
        # augmented
        k = self.U.shape[1]
        Q = np.zeros((k + 1, k + 1))
        Q[:k, :k] = np.diag(self.s)
        Q[:k, k:k + 1] = m
        Q[k, k] = p_norm
        # SVD of Q
        U_q, s_q, Vt_q = _svd(Q, full_matrices=False)
        # update U/s/Vt
        p_normed = p / (p_norm + 1e-30) if p_norm > 1e-30 else np.zeros_like(p)
        U_ext = np.hstack([self.U, p_normed])
        self.U = U_ext @ U_q
        self.s = s_q
        # expand Vt: new row
        Vt_ext = np.zeros((self.Vt.shape[0] + 1, self.Vt.shape[1] + 1))
        Vt_ext[:-1, :-1] = self.Vt
        Vt_ext[-1, -1] = 1.0
        self.Vt = Vt_q @ Vt_ext
        # truncate to rank
        if self.s.size > self.rank:
            self.U = self.U[:, :self.rank]
            self.s = self.s[:self.rank]
            self.Vt = self.Vt[:self.rank]


__all__ = ["IncrementalSVD"]
