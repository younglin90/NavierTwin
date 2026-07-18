"""Hyperparameter 자동 튜닝 — Optuna (선택) / BoTorch (선택) / Random fallback.

검색 공간은 {name: (low, high, type)} 형식, type ∈ {"float", "int", "logfloat"}.

Examples:
    >>> from naviertwin.core.optimization.hyperopt import hyperopt
    >>> def f(params):
    ...     return (params["lr"] - 0.01) ** 2 + (params["n_layers"] - 3) ** 2
    >>> space = {
    ...     "lr": (1e-4, 1e-1, "logfloat"),
    ...     "n_layers": (1, 5, "int"),
    ... }
    >>> best, hist = hyperopt(f, space, n_trials=20, backend="random", seed=0)
    >>> "lr" in best
    True
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _sample_random(
    space: dict[str, tuple],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """한 번의 random 샘플."""
    def sample_item(item: tuple[str, tuple]) -> tuple[str, Any]:
        name, (low, high, typ) = item
        if typ == "float":
            return name, float(rng.uniform(low, high))
        if typ == "logfloat":
            log_lo = np.log10(low)
            log_hi = np.log10(high)
            return name, float(10 ** rng.uniform(log_lo, log_hi))
        if typ == "int":
            return name, int(rng.integers(low, high + 1))
        raise ValueError(f"알 수 없는 type: {typ}")

    return dict(map(sample_item, space.items()))


def _to_bounds(space: dict[str, tuple]) -> NDArray[np.float64]:
    """BoTorch 용 (n, 2) bounds. int 는 그대로 float 영역."""
    def bound_row(item: tuple[str, tuple]) -> list[float]:
        _, (low, high, typ) = item
        if typ == "logfloat":
            return [np.log10(low), np.log10(high)]
        return [low, high]

    return np.asarray(tuple(map(bound_row, space.items())), dtype=np.float64)


def _denormalize(
    x: NDArray[np.float64],
    space: dict[str, tuple],
) -> dict[str, Any]:
    """BoTorch 제안값 x 를 space 형식 dict 로."""
    def denorm_item(pair: tuple[np.float64, tuple[str, tuple]]) -> tuple[str, Any]:
        xi, item = pair
        name, (_, _, typ) = item
        v = float(xi)
        if typ == "logfloat":
            return name, float(10 ** v)
        if typ == "int":
            return name, int(round(v))
        return name, v

    return dict(map(denorm_item, zip(x, space.items(), strict=True)))


def hyperopt(
    objective: Callable[[dict[str, Any]], float],
    space: dict[str, tuple],
    n_trials: int = 20,
    backend: str = "random",
    seed: int | None = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """하이퍼파라미터 최적화.

    Args:
        objective: dict[param] → scalar loss (최소화).
        space: {name: (low, high, type)}.
        n_trials: 시도 횟수.
        backend: "random" / "botorch" / "optuna".
        seed: 재현.

    Returns:
        (best_params, history_list).
    """
    if backend == "optuna":
        return _hyperopt_optuna(objective, space, n_trials, seed)
    if backend == "botorch":
        return _hyperopt_botorch(objective, space, n_trials, seed)
    return _hyperopt_random(objective, space, n_trials, seed)


def _hyperopt_random(
    objective: Callable[[dict[str, Any]], float],
    space: dict[str, tuple],
    n_trials: int,
    seed: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_val = np.inf
    trial_idx = 0
    while trial_idx < n_trials:
        p = _sample_random(space, rng)
        v = float(objective(p))
        history.append({"params": p, "value": v})
        if v < best_val:
            best_val = v
            best = p
        trial_idx += 1
    logger.info("hyperopt(random): %d trials, best=%.6g", n_trials, best_val)
    return best or {}, history


def _hyperopt_botorch(
    objective: Callable[[dict[str, Any]], float],
    space: dict[str, tuple],
    n_trials: int,
    seed: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        from naviertwin.core.optimization.bayesian_opt_botorch import (
            BoTorchBayesianOpt,
        )
    except RuntimeError as exc:
        logger.warning("BoTorch 불가 → random: %s", exc)
        return _hyperopt_random(objective, space, n_trials, seed)

    bounds = _to_bounds(space)
    n_initial = max(3, min(8, n_trials // 3))
    history: list[dict[str, Any]] = []

    def wrapped(x: NDArray[np.float64]) -> float:
        p = _denormalize(x, space)
        v = float(objective(p))
        history.append({"params": p, "value": v})
        return v

    opt = BoTorchBayesianOpt(
        bounds=bounds, n_initial=n_initial,
        max_iter=max(1, n_trials - n_initial), q=1,
        acquisition="qei", seed=seed,
    )
    x_best, f_best = opt.minimize(wrapped)
    best = _denormalize(x_best, space)
    logger.info("hyperopt(botorch): best=%.6g", f_best)
    return best, history


def _hyperopt_optuna(
    objective: Callable[[dict[str, Any]], float],
    space: dict[str, tuple],
    n_trials: int,
    seed: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna 미설치 → random")
        return _hyperopt_random(objective, space, n_trials, seed)

    history: list[dict[str, Any]] = []

    def _obj(trial: "optuna.Trial") -> float:
        def suggest_item(item: tuple[str, tuple]) -> tuple[str, Any]:
            name, (low, high, typ) = item
            if typ == "int":
                return name, trial.suggest_int(name, int(low), int(high))
            if typ == "logfloat":
                value = trial.suggest_float(name, float(low), float(high), log=True)
                return name, value
            return name, trial.suggest_float(name, float(low), float(high))

        p = dict(map(suggest_item, space.items()))
        v = float(objective(p))
        history.append({"params": p, "value": v})
        return v

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)
    logger.info("hyperopt(optuna): best=%.6g", study.best_value)
    return dict(study.best_params), history


__all__ = ["hyperopt"]
