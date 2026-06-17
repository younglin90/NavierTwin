"""Kernel SHAP 경량 explainer (SHAP 미설치 환경에서도 동작).

예측 함수 f(X) 에 대해 각 feature 의 Shapley value 를 무작위 subset 샘플로
추정한다. `shap` 패키지가 있으면 그걸 사용.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.explainability.shap_explainer import KernelSHAP
    >>> def f(X):
    ...     return 3.0 * X[:, 0] + 2.0 * X[:, 1]
    >>> rng = np.random.default_rng(0)
    >>> background = rng.standard_normal((50, 2))
    >>> expl = KernelSHAP(f, background, n_samples=80, seed=0)
    >>> phi = expl.explain(np.array([[1.0, 1.0]]))
    >>> phi.shape
    (1, 2)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class KernelSHAP:
    """경량 Shapley value 추정기.

    각 대상 x 에 대해 feature 부분집합을 무작위로 선택한 뒤 그 안/밖 조건부
    예측을 비교한 평균을 feature 기여도로 추정한다. Monte Carlo Shapley.

    Args:
        f: 모델 예측 함수 (N, d) → (N,).
        background: 배경 샘플 (M, d) — 각 feature 의 "off" 상태 대체.
        n_samples: Monte Carlo 샘플 수 (반복).
        seed: 재현성.
    """

    def __init__(
        self,
        f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        background: NDArray[np.float64],
        n_samples: int = 100,
        seed: int | None = None,
    ) -> None:
        self.f = f
        self.background = np.asarray(background, dtype=np.float64)
        if self.background.ndim != 2:
            raise ValueError("background (M, d) 2D 필요")
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def explain(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """각 샘플 x 에 대해 (d,) 기여도 반환. 총 shape (N, d)."""
        try:
            import shap  # noqa: F401

            # shap 이 있어도 우리 경량 구현 유지 (의존성 선택 가능)
        except ImportError:
            pass

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        N, d = X.shape
        M = self.background.shape[0]
        phi = np.zeros((N, d))

        n = 0
        while n < N:
            x = X[n]
            contrib = np.zeros((d, self.n_samples))
            s = 0
            while s < self.n_samples:
                perm = self.rng.permutation(d)
                # 배경 샘플 하나 선택
                bg = self.background[self.rng.integers(M)]
                # incremental 상태 구축
                state = bg.copy()
                f_prev = float(self.f(state[None, :])[0])
                perm_idx = 0
                while perm_idx < perm.size:
                    j = perm[perm_idx]
                    state[j] = x[j]
                    f_curr = float(self.f(state[None, :])[0])
                    contrib[j, s] = f_curr - f_prev
                    f_prev = f_curr
                    perm_idx += 1
                s += 1
            phi[n] = contrib.mean(axis=1)
            n += 1
        logger.info("KernelSHAP explain: N=%d, d=%d, samples=%d", N, d, self.n_samples)
        return phi


__all__ = ["KernelSHAP"]
