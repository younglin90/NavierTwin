"""Successive halving — 한정 자원하에 하이퍼파라미터 탐색.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.halving import successive_halving
    >>> configs = list(map(lambda v: {"x": v}, np.linspace(-2, 2, 16)))
    >>> # budget = max iter (eval 은 budget 에 의존)
    >>> def evaluator(cfg, budget): return -(cfg["x"] - 1) ** 2 * budget
    >>> best, hist = successive_halving(configs, evaluator, max_budget=100, eta=4)
    >>> abs(best["x"] - 1) < 1.0
    True
"""

from __future__ import annotations

from typing import Any, Callable


def successive_halving(
    configs: list[dict[str, Any]],
    evaluator: Callable[[dict[str, Any], int], float],
    *, max_budget: int = 100, eta: int = 3,
    higher_better: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """configs 를 점진적으로 줄여가며 budget 늘림."""
    survivors = list(configs)
    budget = max(1, max_budget // (eta ** max(0, _n_stages(len(configs), eta) - 1)))
    history: list[dict[str, Any]] = []
    while True:
        scored = []
        idx = 0
        while idx < len(survivors):
            cfg = survivors[idx]
            s = float(evaluator(cfg, budget))
            scored.append((s, cfg))
            history.append({"config": cfg, "budget": budget, "score": s})
            idx += 1
        scored.sort(key=lambda t: -t[0] if higher_better else t[0])
        keep = max(1, len(scored) // eta)
        survivors = []
        idx = 0
        while idx < keep:
            survivors.append(scored[idx][1])
            idx += 1
        budget = min(max_budget, budget * eta)
        if len(survivors) == 1 or budget >= max_budget:
            # final evaluation
            if len(survivors) > 1:
                scored = []
                idx = 0
                while idx < len(survivors):
                    cfg = survivors[idx]
                    scored.append((float(evaluator(cfg, max_budget)), cfg))
                    idx += 1
                scored = sorted(scored, key=lambda t: -t[0] if higher_better else t[0])
                survivors = [scored[0][1]]
            break
    return survivors[0], history


def _n_stages(n: int, eta: int) -> int:
    s = 0
    while n > 1:
        n = max(1, n // eta)
        s += 1
    return max(1, s)


__all__ = ["successive_halving"]
