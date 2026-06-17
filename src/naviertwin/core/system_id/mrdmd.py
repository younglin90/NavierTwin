"""mrDMD — Multi-resolution DMD (Kutz et al. 2016).

시간 윈도우를 재귀적으로 분할하여 각 레벨에서 slow modes 추출.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.system_id.mrdmd import mrdmd
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 64))
    >>> tree = mrdmd(X, max_levels=2, rank_per_level=2)
    >>> len(tree)
    7
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray


def _exact_dmd(
    X: NDArray[np.float64],
    rank: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    U, s, Vt = _svd(X1, full_matrices=False)
    r = min(rank, U.shape[1])
    U = U[:, :r]
    s = s[:r]
    Vt = Vt[:r]
    A_tilde = U.T @ X2 @ Vt.T @ np.diag(1.0 / (s + 1e-30))
    evals, W = np.linalg.eig(A_tilde)
    modes = X2 @ Vt.T @ np.diag(1.0 / (s + 1e-30)) @ W
    return evals, modes


def mrdmd(
    X: NDArray[np.float64],
    *,
    max_levels: int = 3,
    rank_per_level: int = 3,
    slow_threshold: float = 0.5,
) -> list[dict]:
    """재귀 mrDMD. 반환: 각 노드 dict {level, t_start, t_end, evals, modes}."""
    X = np.asarray(X, dtype=np.float64)
    nodes: list[dict] = []

    def recurse(seg: NDArray, level: int, t0: int, t1: int) -> None:
        if seg.shape[1] < 4:
            return
        evals, modes = _exact_dmd(seg, rank_per_level)
        # frequency: |log(λ)| / dt; here dt=1, so just |log λ|
        freqs = np.abs(np.log(evals + 1e-30))
        slow_mask = freqs < slow_threshold
        nodes.append({
            "level": level,
            "t_start": t0,
            "t_end": t1,
            "evals": evals[slow_mask],
            "modes": modes[:, slow_mask],
        })
        if level >= max_levels:
            return
        mid = seg.shape[1] // 2
        recurse(seg[:, :mid], level + 1, t0, t0 + mid)
        recurse(seg[:, mid:], level + 1, t0 + mid, t1)

    recurse(X, 0, 0, X.shape[1])
    return nodes


__all__ = ["mrdmd"]
