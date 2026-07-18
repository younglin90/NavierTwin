"""BoTorch 기반 고급 Bayesian Optimization — qEI / UCB + batch parallel.

기존 `bayesian_opt.py` 는 sklearn GP + 단일 후보. 이 모듈은:
    - BoTorch SingleTaskGP (GPU 지원)
    - qExpectedImprovement (배치 q > 1 병렬 획득)
    - UpperConfidenceBound 선택
    - 자동 fit with MLL

BoTorch 미설치 시 RuntimeError (폴백은 기존 `bayesian_opt.py` 가 담당).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.bayesian_opt_botorch import BoTorchBayesianOpt
    >>>
    >>> def obj(x):
    ...     return float((x[0] - 0.3) ** 2 + (x[1] + 0.2) ** 2)
    >>>
    >>> opt = BoTorchBayesianOpt(
    ...     bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
    ...     n_initial=5, max_iter=10, q=2,
    ...     acquisition="qei", seed=0,
    ... )
    >>> x_best, f_best = opt.minimize(obj)
    >>> f_best < 0.3
    True
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _require_botorch() -> None:
    try:
        import botorch  # noqa: F401
        import gpytorch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "botorch + gpytorch 필요: pip install botorch"
        ) from exc


class BoTorchBayesianOpt:
    """BoTorch SingleTaskGP + qEI/UCB 배치 BO.

    Args:
        bounds: (n_dims, 2).
        n_initial: 초기 랜덤 평가.
        max_iter: BO 반복 수.
        q: 배치 크기 (병렬 후보). 1 이면 단일.
        acquisition: "qei" / "ucb".
        ucb_beta: UCB β 파라미터.
        num_restarts / raw_samples: AF 최적화 옵션.
        seed: 재현.
    """

    def __init__(
        self,
        bounds: NDArray[np.float64],
        n_initial: int = 8,
        max_iter: int = 20,
        q: int = 1,
        acquisition: str = "qei",
        ucb_beta: float = 2.0,
        num_restarts: int = 5,
        raw_samples: int = 64,
        seed: int | None = None,
    ) -> None:
        _require_botorch()
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"bounds shape={bounds.shape} != (n, 2)")
        self.bounds = bounds
        self.n_initial = n_initial
        self.max_iter = max_iter
        self.q = q
        self.acquisition = acquisition
        self.ucb_beta = ucb_beta
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.seed = seed

        self.X_: list[NDArray[np.float64]] = []
        self.y_: list[float] = []

    def _sample(
        self, n: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        return lows + rng.random((n, self.bounds.shape[0])) * (highs - lows)

    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> Any:
        """BoTorch GP 모델 학습."""
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood

        X_t = torch.tensor(X, dtype=torch.float64)
        # 최소화 → BoTorch 는 최대화이므로 부호 반전
        y_t = torch.tensor(-y.reshape(-1, 1), dtype=torch.float64)
        gp = SingleTaskGP(X_t, y_t)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:  # noqa: BLE001
            logger.warning("GP 학습 경고: %s", e)
        return gp

    def _optimize_af(
        self, gp: Any, best_f: float
    ) -> NDArray[np.float64]:
        import torch
        from botorch.acquisition import (
            UpperConfidenceBound,
            qExpectedImprovement,
        )
        from botorch.optim import optimize_acqf

        bounds_t = torch.tensor(self.bounds.T, dtype=torch.float64)

        if self.acquisition == "qei":
            # best_f 는 최대화 기준 (이미 부호 반전 됨)
            af = qExpectedImprovement(gp, best_f=-best_f)
        elif self.acquisition == "ucb":
            af = UpperConfidenceBound(gp, beta=self.ucb_beta)
        else:
            raise ValueError(f"알 수 없는 acquisition: {self.acquisition}")

        X_cand, _ = optimize_acqf(
            af,
            bounds=bounds_t,
            q=self.q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return X_cand.detach().cpu().numpy()

    def minimize(
        self, f: Callable[[NDArray[np.float64]], float]
    ) -> tuple[NDArray[np.float64], float]:
        import torch

        if self.seed is not None:
            torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        X_init = self._sample(self.n_initial, rng)
        init_idx = 0
        while init_idx < X_init.shape[0]:
            x = X_init[init_idx]
            self.X_.append(x)
            self.y_.append(float(f(x)))
            init_idx += 1

        iter_idx = 0
        while iter_idx < self.max_iter:
            X_arr = np.vstack(self.X_)
            y_arr = np.asarray(self.y_, dtype=np.float64)
            gp = self._fit_gp(X_arr, y_arr)
            X_cand = self._optimize_af(gp, float(np.min(y_arr)))
            cand_idx = 0
            while cand_idx < X_cand.shape[0]:
                x_new = X_cand[cand_idx]
                y_new = float(f(x_new))
                self.X_.append(x_new)
                self.y_.append(y_new)
                cand_idx += 1
            iter_idx += 1

        y_arr = np.asarray(self.y_, dtype=np.float64)
        idx = int(np.argmin(y_arr))
        logger.info(
            "BoTorch BO 완료: n_eval=%d, f_best=%.6g, acq=%s, q=%d",
            len(self.y_), float(y_arr[idx]), self.acquisition, self.q,
        )
        return self.X_[idx], float(y_arr[idx])


__all__ = ["BoTorchBayesianOpt"]
