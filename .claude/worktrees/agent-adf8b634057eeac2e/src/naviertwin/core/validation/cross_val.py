"""Cross-validation + grid search (no sklearn dependency).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.validation.cross_val import kfold_scores
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.preprocessing.splitter import k_fold_indices


def kfold_scores(
    X: NDArray[np.float64], y: NDArray[np.float64],
    fit_predict: Callable[[NDArray, NDArray, NDArray], NDArray],
    score_fn: Callable[[NDArray, NDArray], float],
    *, k: int = 5, seed: int | None = 0,
) -> list[float]:
    """fit_predict(X_tr, y_tr, X_val) → y_pred.  반환: k개 fold 점수."""
    def _score(split: tuple[NDArray[np.int64], NDArray[np.int64]]) -> float:
        tr_idx, val_idx = split
        y_pred = fit_predict(X[tr_idx], y[tr_idx], X[val_idx])
        return float(score_fn(y[val_idx], y_pred))

    return list(map(_score, k_fold_indices(X.shape[0], k, seed=seed)))


def grid_search(
    X: NDArray[np.float64], y: NDArray[np.float64],
    fit_predict_factory: Callable[[dict], Callable],
    param_grid: dict[str, list[Any]],
    score_fn: Callable[[NDArray, NDArray], float],
    *, k: int = 3, seed: int | None = 0, higher_better: bool = False,
) -> dict:
    keys = list(param_grid.keys())
    best: dict = {"score": np.inf if not higher_better else -np.inf, "params": None}

    def _evaluate(vals: tuple[Any, ...]) -> dict:
        params = dict(zip(keys, vals, strict=True))
        fp = fit_predict_factory(params)
        scores = kfold_scores(X, y, fp, score_fn, k=k, seed=seed)
        mean_s = float(np.mean(scores))
        return {"params": params, "mean_score": mean_s, "scores": scores}

    param_values = tuple(map(param_grid.__getitem__, keys))
    history = list(map(_evaluate, product(*param_values)))
    if history:
        chosen = max(history, key=lambda row: row["mean_score"]) if higher_better else min(
            history, key=lambda row: row["mean_score"]
        )
        best = {"score": chosen["mean_score"], "params": chosen["params"]}
    return {"best": best, "history": history}


__all__ = ["kfold_scores", "grid_search"]
