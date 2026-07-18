"""Active Learning — 불확실성 기반 쿼리 선택.

pool 에서 다음 평가할 샘플을 고르는 2 가지 전략:
    - "variance": GP predict 의 std 가 가장 큰 점
    - "random": 단순 균일 샘플링 (베이스라인)

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.online_learning.active_learning import (
    ...     select_next_samples,
    ... )
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 2))
    >>> y = np.sin(X.sum(axis=1))
    >>> gp = GaussianProcessRegressor()
    >>> gp.fit(X, y)
    >>> pool = rng.standard_normal((100, 2))
    >>> idx = select_next_samples(gp, pool, k=5, strategy="variance")
    >>> len(idx)
    5
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def select_next_samples(
    model: object,
    pool: NDArray[np.float64],
    k: int = 1,
    strategy: str = "variance",
    seed: int | None = None,
) -> NDArray[np.int64]:
    """pool 에서 다음 평가할 k 개 인덱스 선택."""
    pool = np.asarray(pool, dtype=np.float64)
    if pool.ndim != 2:
        raise ValueError(f"pool (N, d) 2D 필요: {pool.shape}")
    if k <= 0 or k > pool.shape[0]:
        raise ValueError(f"k 는 1..{pool.shape[0]} 범위")

    rng = np.random.default_rng(seed)

    if strategy == "random":
        return rng.choice(pool.shape[0], size=k, replace=False)
    if strategy == "variance":
        try:
            _, std = model.predict(pool, return_std=True)
        except TypeError:
            # GP 아닌 경우 — 랜덤 fallback
            logger.warning("모델이 return_std 를 지원하지 않음 — random fallback")
            return rng.choice(pool.shape[0], size=k, replace=False)
        return np.argsort(-std)[:k]

    raise ValueError(f"알 수 없는 strategy: {strategy}")


def active_loop(
    model_factory: "callable",
    X_init: NDArray[np.float64],
    y_init: NDArray[np.float64],
    pool: NDArray[np.float64],
    oracle: "callable",
    n_query: int = 10,
    strategy: str = "variance",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """간단한 pool-based active loop.

    Args:
        model_factory: 매 iter 새 모델 인스턴스를 생성해 반환.
        X_init, y_init: 초기 라벨 데이터.
        pool: 라벨 후보 풀.
        oracle: pool[i] 에 대한 실제 라벨을 반환하는 함수.
        n_query: 추가로 라벨할 샘플 수.
        strategy: "variance" / "random".

    Returns:
        확장된 (X, y).
    """
    X = np.array(X_init, dtype=np.float64)
    y = np.array(y_init, dtype=np.float64)
    mask = np.ones(len(pool), dtype=bool)
    query = 0
    while query < n_query:
        model = model_factory()
        model.fit(X, y)
        remaining = np.where(mask)[0]
        sub_idx = select_next_samples(model, pool[remaining], k=1, strategy=strategy)
        pick = remaining[sub_idx[0]]
        mask[pick] = False
        y_new = oracle(pool[pick])
        X = np.vstack([X, pool[pick : pick + 1]])
        y = np.append(y, y_new)
        query += 1
    logger.info("active_loop 종료: 총 %d 샘플", len(X))
    return X, y


__all__ = ["select_next_samples", "active_loop"]
