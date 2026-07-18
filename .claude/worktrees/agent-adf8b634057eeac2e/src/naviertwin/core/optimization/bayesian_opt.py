"""Bayesian Optimization — GP surrogate + Expected Improvement.

scikit-learn GaussianProcessRegressor 기반 경량 구현.
scikit-optimize / NLopt 없이도 동작 (core 의존성만 사용).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.bayesian_opt import BayesianOptimizer
    >>>
    >>> def objective(x):
    ...     return float((x[0] - 0.3) ** 2 + (x[1] + 0.5) ** 2)
    >>>
    >>> opt = BayesianOptimizer(
    ...     bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
    ...     n_initial=8, max_iter=10, seed=0,
    ... )
    >>> x_best, f_best = opt.minimize(objective)
    >>> x_best.shape
    (2,)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class BayesianOptimizer:
    """GP + Expected Improvement 최소화기.

    Args:
        bounds: (n_dims, 2) 파라미터 경계.
        n_initial: 초기 랜덤 평가 수.
        max_iter: BO 반복 수.
        xi: EI exploration-exploitation 파라미터.
        n_candidates: 각 iter 당 AF 최대화용 랜덤 후보 수.
        seed: 재현용 seed.
    """

    def __init__(
        self,
        bounds: NDArray[np.float64],
        n_initial: int = 5,
        max_iter: int = 20,
        xi: float = 0.01,
        n_candidates: int = 2048,
        seed: int | None = None,
    ) -> None:
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"bounds shape={bounds.shape} != (n, 2)")
        self.bounds = bounds
        self.n_initial = n_initial
        self.max_iter = max_iter
        self.xi = xi
        self.n_candidates = n_candidates
        self.seed = seed

        self.X_: list[NDArray[np.float64]] = []
        self.y_: list[float] = []

    def _sample(
        self, n: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        return lows + rng.random((n, self.bounds.shape[0])) * (highs - lows)

    def _ei(
        self,
        X: NDArray[np.float64],
        gp: "object",
        f_best: float,
    ) -> NDArray[np.float64]:
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        gamma = (f_best - mu - self.xi) / sigma
        return (f_best - mu - self.xi) * norm.cdf(gamma) + sigma * norm.pdf(gamma)

    def minimize(
        self, f: Callable[[NDArray[np.float64]], float]
    ) -> tuple[NDArray[np.float64], float]:
        """f(x) 를 최소화하는 x 를 반환한다."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn 필요: pip install scikit-learn"
            ) from exc

        rng = np.random.default_rng(self.seed)
        X_init = self._sample(self.n_initial, rng)
        init_idx = 0
        while init_idx < X_init.shape[0]:
            x = X_init[init_idx]
            val = float(f(x))
            self.X_.append(x)
            self.y_.append(val)
            init_idx += 1

        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)

        step = 0
        while step < self.max_iter:
            X_arr = np.vstack(self.X_)
            y_arr = np.asarray(self.y_, dtype=np.float64)
            gp = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, n_restarts_optimizer=2
            )
            gp.fit(X_arr, y_arr)

            cands = self._sample(self.n_candidates, rng)
            ei = self._ei(cands, gp, float(np.min(y_arr)))
            x_next = cands[int(np.argmax(ei))]
            y_next = float(f(x_next))

            self.X_.append(x_next)
            self.y_.append(y_next)
            step += 1

        y_arr = np.asarray(self.y_, dtype=np.float64)
        best_idx = int(np.argmin(y_arr))
        logger.info(
            "BO 완료: n_eval=%d, f_best=%.6g",
            len(self.y_),
            float(y_arr[best_idx]),
        )
        return self.X_[best_idx], float(y_arr[best_idx])


__all__ = ["BayesianOptimizer"]
