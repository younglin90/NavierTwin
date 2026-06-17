"""NSGA-II 다목적 최적화 — NumPy 직접 구현 (pygmo 없이).

References:
    Deb et al., "A fast and elitist multiobjective genetic algorithm: NSGA-II", 2002.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.moo_optimizer import NSGA2
    >>> def obj(x):
    ...     f1 = float(np.sum((x - 0.2) ** 2))
    ...     f2 = float(np.sum((x + 0.5) ** 2))
    ...     return [f1, f2]
    >>> nsga = NSGA2(bounds=np.array([[-1, 1]] * 2), n_obj=2, pop_size=20, n_gen=10, seed=0)
    >>> pareto, objs = nsga.optimize(obj)
    >>> pareto.shape[1] == 2 and objs.shape[1] == 2
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _dominates(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    """a 가 b 를 (최소화 기준) 지배하는지."""
    return bool(np.all(a <= b) and np.any(a < b))


def _fast_non_dominated_sort(F: NDArray[np.float64]) -> list[list[int]]:
    """F: (N, m) → front 별 인덱스 리스트."""
    N = F.shape[0]
    S: list[list[int]] = list(map(lambda _: [], range(N)))
    n = np.zeros(N, dtype=int)
    rank = np.zeros(N, dtype=int)
    fronts: list[list[int]] = [[]]
    p = 0
    while p < N:
        q = 0
        while q < N:
            if p == q:
                q += 1
                continue
            if _dominates(F[p], F[q]):
                S[p].append(q)
            elif _dominates(F[q], F[p]):
                n[p] += 1
            q += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
        p += 1
    i = 0
    while fronts[i]:
        nxt: list[int] = []
        front_pos = 0
        while front_pos < len(fronts[i]):
            p = fronts[i][front_pos]
            dominated_pos = 0
            while dominated_pos < len(S[p]):
                q = S[p][dominated_pos]
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    nxt.append(q)
                dominated_pos += 1
            front_pos += 1
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def _crowding_distance(F: NDArray[np.float64], idx: list[int]) -> NDArray[np.float64]:
    """front 내 밀집도."""
    m = F.shape[1]
    dist = np.zeros(len(idx))
    if len(idx) <= 2:
        dist[:] = np.inf
        return dist
    sub = F[idx]
    k = 0
    while k < m:
        order = np.argsort(sub[:, k])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        fmin, fmax = sub[order[0], k], sub[order[-1], k]
        if fmax - fmin == 0:
            k += 1
            continue
        i = 1
        while i < len(idx) - 1:
            dist[order[i]] += (sub[order[i + 1], k] - sub[order[i - 1], k]) / (fmax - fmin)
            i += 1
        k += 1
    return dist


class NSGA2:
    """기본 NSGA-II — SBX 교차 + 다항식 변이."""

    def __init__(
        self,
        bounds: NDArray[np.float64],
        n_obj: int,
        pop_size: int = 50,
        n_gen: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.n_obj = n_obj
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.rng = np.random.default_rng(seed)

    def _init_pop(self) -> NDArray[np.float64]:
        lows, highs = self.bounds[:, 0], self.bounds[:, 1]
        return lows + self.rng.random((self.pop_size, len(lows))) * (highs - lows)

    def _crossover(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.rng.random() > self.crossover_prob:
            return a.copy(), b.copy()
        alpha = self.rng.random(a.size)
        c1 = alpha * a + (1 - alpha) * b
        c2 = (1 - alpha) * a + alpha * b
        return c1, c2

    def _mutate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        lows, highs = self.bounds[:, 0], self.bounds[:, 1]
        mask = self.rng.random(x.size) < self.mutation_prob
        noise = self.rng.standard_normal(x.size) * 0.1 * (highs - lows)
        x2 = np.where(mask, x + noise, x)
        return np.clip(x2, lows, highs)

    def optimize(
        self, objective: Callable[[NDArray[np.float64]], list[float]]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        P = self._init_pop()
        F = np.array(list(map(objective, P)))

        gen = 0
        while gen < self.n_gen:
            # 자손 생성
            Q = np.zeros_like(P)
            i = 0
            while i < self.pop_size:
                a, b = P[self.rng.integers(self.pop_size, size=2)]
                c1, c2 = self._crossover(a, b)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                Q[i] = c1
                if i + 1 < self.pop_size:
                    Q[i + 1] = c2
                i += 2
            FQ = np.array(list(map(objective, Q)))

            # P+Q 에서 비지배 정렬
            R = np.vstack([P, Q])
            FR = np.vstack([F, FQ])
            fronts = _fast_non_dominated_sort(FR)

            new_idx: list[int] = []
            front_idx = 0
            while front_idx < len(fronts):
                front = fronts[front_idx]
                if len(new_idx) + len(front) <= self.pop_size:
                    new_idx.extend(front)
                else:
                    dist = _crowding_distance(FR, front)
                    order = np.argsort(-dist)
                    order_idx = 0
                    while order_idx < len(order):
                        k = order[order_idx]
                        if len(new_idx) < self.pop_size:
                            new_idx.append(front[k])
                        order_idx += 1
                    break
                front_idx += 1
            P = R[new_idx]
            F = FR[new_idx]
            gen += 1

        # 최종 첫 번째 프론트
        fronts = _fast_non_dominated_sort(F)
        pareto_idx = fronts[0]
        logger.info("NSGA-II 완료: 파레토 전선 크기 %d", len(pareto_idx))
        return P[pareto_idx], F[pareto_idx]


__all__ = ["NSGA2"]
