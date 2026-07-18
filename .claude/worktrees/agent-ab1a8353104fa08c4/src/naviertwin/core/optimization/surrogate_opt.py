"""Surrogate-based Optimization (SBO) — RBF/Kriging + scipy 로컬 최적화.

매 반복:
    1. 현재 샘플로 surrogate 피팅
    2. surrogate 위 최소점 탐색 (scipy.optimize)
    3. 그 점에서 실제 f 평가 → 샘플에 추가

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.surrogate_opt import SurrogateOptimizer
    >>> def f(x):
    ...     return float((x[0] - 0.3) ** 2 + (x[1] + 0.2) ** 2)
    >>> opt = SurrogateOptimizer(
    ...     bounds=np.array([[-1, 1], [-1, 1]]), surrogate_kind="rbf",
    ...     n_initial=5, max_iter=10, seed=0,
    ... )
    >>> x_best, f_best = opt.minimize(f)
    >>> f_best < 0.3
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class SurrogateOptimizer:
    """RBF surrogate 기반 순차 최소화."""

    def __init__(
        self,
        bounds: NDArray[np.float64],
        surrogate_kind: str = "rbf",
        n_initial: int = 8,
        max_iter: int = 20,
        seed: int | None = None,
    ) -> None:
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.surrogate_kind = surrogate_kind
        self.n_initial = n_initial
        self.max_iter = max_iter
        self.seed = seed

        self.X_: list[NDArray[np.float64]] = []
        self.y_: list[float] = []

    def _sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        return lows + rng.random((n, self.bounds.shape[0])) * (highs - lows)

    def _fit_surrogate(self) -> object:
        if self.surrogate_kind == "rbf":
            from naviertwin.core.surrogate.rbf_surrogate import RBFSurrogate

            sur = RBFSurrogate()
        else:
            from naviertwin.core.surrogate.kriging_surrogate import KrigingSurrogate

            sur = KrigingSurrogate()
        X = np.vstack(self.X_)
        y = np.asarray(self.y_, dtype=np.float64).reshape(-1, 1)
        sur.fit(X, y)
        return sur

    def minimize(
        self, f: Callable[[NDArray[np.float64]], float]
    ) -> tuple[NDArray[np.float64], float]:
        rng = np.random.default_rng(self.seed)
        initial = self._sample(self.n_initial, rng)
        self.X_.extend(initial)
        self.y_.extend(map(lambda x: float(f(x)), initial))

        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]

        iter_idx = 0
        while iter_idx < self.max_iter:
            sur = self._fit_surrogate()

            def surrogate_val(x: np.ndarray) -> float:
                return float(sur.predict(np.asarray(x).reshape(1, -1))[0])

            # 다중 시작점 로컬 탐색
            starts = self._sample(5, rng)
            results = tuple(
                map(
                    lambda x0: minimize(
                        surrogate_val,
                        x0,
                        method="L-BFGS-B",
                        bounds=list(zip(lows, highs)),
                    ),
                    starts,
                )
            )
            vals = np.fromiter(
                map(lambda res: float(res.fun), results), dtype=np.float64, count=len(results)
            )
            best_x = results[int(np.argmin(vals))].x
            y_new = float(f(best_x))
            self.X_.append(best_x)
            self.y_.append(y_new)
            iter_idx += 1

        y_arr = np.asarray(self.y_, dtype=np.float64)
        idx = int(np.argmin(y_arr))
        logger.info(
            "SBO 완료: n_eval=%d, f_best=%.6g", len(self.y_), float(y_arr[idx])
        )
        return self.X_[idx], float(y_arr[idx])


__all__ = ["SurrogateOptimizer"]
