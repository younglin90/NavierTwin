"""Pipeline 자동 튜닝 — n_modes / surrogate kernel 을 hyperopt 로 최적화.

NavierTwinPipeline 의 (reducer + surrogate) 를 주어진 snapshots/params 에 대해
train/validation 분할 → RMSE 최소화로 자동 선택.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline
    >>> from naviertwin.core.digital_twin.pipeline_hyperopt import auto_tune_pipeline
    >>> # rng = np.random.default_rng(0)
    >>> # X = rng.standard_normal((20, 12))
    >>> # params = np.linspace(0, 1, 12).reshape(-1, 1)
    >>> # best = auto_tune_pipeline(X, params, n_trials=6)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _build_and_eval(
    snapshots: NDArray[np.float64],
    params: NDArray[np.float64],
    n_modes: int,
    surrogate_kind: str,
    val_ratio: float,
    seed: int,
) -> float:
    """주어진 하이퍼파라미터로 파이프라인을 학습/평가 → validation RMSE."""
    from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

    n = snapshots.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(2, int(round(n * val_ratio)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    pipe = NavierTwinPipeline(
        reducer_kind="pod",
        n_modes=int(max(1, min(n_modes, len(train_idx) - 1))),
        surrogate_kind=surrogate_kind,
    )
    pipe.load_snapshots(snapshots[:, train_idx], field_name="U")
    pipe.reduce()
    pipe.fit_surrogate(params[train_idx])

    # validation: encode 실제 snapshot → compare surrogate 예측 coeffs
    X_val = snapshots[:, val_idx]
    c_true = pipe.state.reducer.encode(X_val)
    c_pred = pipe.state.surrogate.predict(params[val_idx])
    diff = c_true - c_pred
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return rmse


def auto_tune_pipeline(
    snapshots: NDArray[np.float64],
    params: NDArray[np.float64],
    *,
    n_modes_range: tuple[int, int] = (1, 10),
    surrogate_kinds: tuple[str, ...] = ("kriging", "rbf"),
    n_trials: int = 10,
    val_ratio: float = 0.25,
    backend: str = "random",
    seed: int = 0,
) -> dict[str, Any]:
    """n_modes + surrogate_kind 를 hyperopt 로 자동 선택.

    Args:
        snapshots: (n_features, n_snapshots).
        params: (n_snapshots, n_params).
        n_modes_range: 탐색할 n_modes 범위 [low, high].
        surrogate_kinds: 후보 surrogate (kriging/rbf).
        n_trials: hyperopt 시도 횟수.
        val_ratio: validation 분할 비율.
        backend: "random" / "optuna" / "botorch".
        seed: 재현성.

    Returns:
        {"best_params": {...}, "best_rmse": float, "history": [...]}
    """
    from naviertwin.core.optimization.hyperopt import hyperopt

    lo, hi = n_modes_range
    hi = int(min(hi, snapshots.shape[1] - max(2, int(snapshots.shape[1] * val_ratio)) - 1))
    hi = max(hi, lo)

    # surrogate_kind 를 int index 로 인코딩
    space = {
        "n_modes": (lo, hi, "int"),
        "surrogate_idx": (0, len(surrogate_kinds) - 1, "int"),
    }

    def objective(p: dict[str, Any]) -> float:
        kind = surrogate_kinds[int(p["surrogate_idx"])]
        try:
            return _build_and_eval(
                snapshots, params,
                n_modes=int(p["n_modes"]),
                surrogate_kind=kind,
                val_ratio=val_ratio,
                seed=seed,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("hyperopt 시도 실패 %s → penalty: %s", p, e)
            return 1e6

    best, history = hyperopt(
        objective, space, n_trials=n_trials, backend=backend, seed=seed
    )
    best_kind = surrogate_kinds[int(best.get("surrogate_idx", 0))]
    best_params = {
        "n_modes": int(best.get("n_modes", lo)),
        "surrogate_kind": best_kind,
    }
    best_rmse = min(map(lambda h: h["value"], history), default=float("inf"))
    logger.info(
        "auto_tune: best=%s, rmse=%.6g (trials=%d)",
        best_params, best_rmse, len(history),
    )
    return {
        "best_params": best_params,
        "best_rmse": best_rmse,
        "history": history,
    }


__all__ = ["auto_tune_pipeline"]
