"""다목적 최적화 — Pareto front / 비우월 정렬 / hypervolume (2D).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.pareto import pareto_mask
    >>> F = np.array([[1., 5.], [2., 3.], [3., 4.], [5., 1.]])
    >>> pareto_mask(F).tolist()
    [True, True, False, True]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def pareto_mask(F: NDArray[np.float64]) -> NDArray[np.bool_]:
    """minimize 기준 비우월 집합 마스크."""
    if _kernels is None:
        raise ImportError("naviertwin._native._kernels is required by pareto_mask")
    F = np.asarray(F, dtype=np.float64)
    return _kernels.pareto_front(F)


def nondominated_sort(F: NDArray[np.float64]) -> list[NDArray[np.int64]]:
    """NSGA-II 스타일 프론트 리스트."""
    F = np.asarray(F, dtype=np.float64)
    n = F.shape[0]
    S: list[set] = []
    idx = 0
    while idx < n:
        S.append(set())
        idx += 1
    n_count = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]
    p = 0
    while p < n:
        q = 0
        while q < n:
            if p == q:
                q += 1
                continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                S[p].add(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                n_count[p] += 1
            q += 1
        if n_count[p] == 0:
            fronts[0].append(p)
        p += 1
    i = 0
    while fronts[i]:
        next_front: list[int] = []
        p_idx = 0
        while p_idx < len(fronts[i]):
            p = fronts[i][p_idx]
            values = list(S[p])
            q_idx = 0
            while q_idx < len(values):
                q = values[q_idx]
                n_count[q] -= 1
                if n_count[q] == 0:
                    next_front.append(q)
                q_idx += 1
            p_idx += 1
        i += 1
        fronts.append(next_front)
    fronts.pop()  # drop last empty
    out = []
    idx = 0
    while idx < len(fronts):
        out.append(np.asarray(fronts[idx], dtype=np.int64))
        idx += 1
    return out


def hypervolume_2d(
    F: NDArray[np.float64], ref: tuple[float, float],
) -> float:
    """2D 하이퍼볼륨 (minimization, ref=upper-right)."""
    F = np.asarray(F, dtype=np.float64)
    mask = pareto_mask(F)
    front = F[mask]
    if front.size == 0:
        return 0.0
    # 정렬 후 계단 면적
    order = np.argsort(front[:, 0])
    front = front[order]
    hv = 0.0
    y_prev = ref[1]
    idx = 0
    while idx < front.shape[0]:
        p = front[idx]
        if p[1] < y_prev:
            hv += (ref[0] - p[0]) * (y_prev - p[1])
            y_prev = p[1]
        idx += 1
    return float(hv)


__all__ = ["pareto_mask", "nondominated_sort", "hypervolume_2d"]
