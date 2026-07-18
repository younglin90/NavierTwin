"""NSGA-II — constrained-domination 변형 (Deb et al. 2002).

제약 조건 위반 시 violation 합으로 ranking.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.nsga2_constrained import nsga2_constrained
    >>> def obj(x): return np.array([x[0]**2, (x[0]-2)**2])
    >>> def cons(x): return np.array([x[0] - 1.0])  # x ≥ 1
    >>> rng = np.random.default_rng(0)
    >>> pop = nsga2_constrained(obj, cons, n_pop=20, n_gen=10, dim=1,
    ...                           bounds=(np.array([0.0]), np.array([3.0])), rng=rng)
    >>> pop.shape[0]
    20
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def _violation(c: NDArray) -> float:
    return float(np.maximum(0.0, -c).sum())


def _constrained_dominate(a_obj, a_v, b_obj, b_v) -> bool:
    """a constraint-dominates b ?"""
    if a_v < b_v:
        return True
    if a_v > b_v:
        return False
    # both feasible (or equal violation): standard Pareto
    return (a_obj <= b_obj).all() and (a_obj < b_obj).any()


def nsga2_constrained(
    objectives: Callable[[NDArray], NDArray],
    constraints: Callable[[NDArray], NDArray],
    *,
    n_pop: int = 20,
    n_gen: int = 30,
    dim: int = 1,
    bounds: tuple[NDArray, NDArray],
    rng: np.random.Generator | None = None,
) -> NDArray:
    rng = rng if rng is not None else np.random.default_rng(0)
    lo, hi = bounds
    pop = rng.uniform(lo, hi, size=(n_pop, dim))
    gen = 0
    while gen < n_gen:
        # offspring via blend
        idx = rng.integers(0, n_pop, n_pop)
        kids = 0.5 * (pop + pop[idx]) + 0.1 * rng.standard_normal((n_pop, dim))
        kids = np.clip(kids, lo, hi)
        big = np.vstack([pop, kids])
        objs = np.array(list(map(objectives, big)))
        viols = np.array(list(map(lambda x: _violation(constraints(x)), big)))
        # sort by (violation asc, then dominance score)
        scores = np.zeros(len(big))
        i = 0
        while i < len(big):
            j = 0
            while j < len(big):
                if i != j and _constrained_dominate(objs[j], viols[j], objs[i], viols[i]):
                    scores[i] += 1
                j += 1
            i += 1
        order = np.lexsort((scores, viols))
        pop = big[order[:n_pop]]
        gen += 1
    return pop


__all__ = ["nsga2_constrained"]
