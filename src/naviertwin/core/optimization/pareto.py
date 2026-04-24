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


def pareto_mask(F: NDArray[np.float64]) -> NDArray[np.bool_]:
    """minimize 기준 비우월 집합 마스크."""
    F = np.asarray(F, dtype=np.float64)
    n = F.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i ?
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                mask[i] = False
                break
    return mask


def nondominated_sort(F: NDArray[np.float64]) -> list[NDArray[np.int64]]:
    """NSGA-II 스타일 프론트 리스트."""
    F = np.asarray(F, dtype=np.float64)
    n = F.shape[0]
    S: list[set] = [set() for _ in range(n)]
    n_count = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                S[p].add(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                n_count[p] += 1
        if n_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_count[q] -= 1
                if n_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # drop last empty
    return [np.asarray(f, dtype=np.int64) for f in fronts]


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
    for p in front:
        if p[1] < y_prev:
            hv += (ref[0] - p[0]) * (y_prev - p[1])
            y_prev = p[1]
    return float(hv)


__all__ = ["pareto_mask", "nondominated_sort", "hypervolume_2d"]
