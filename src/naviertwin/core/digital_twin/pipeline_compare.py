"""여러 (reducer, surrogate) 조합을 같은 데이터에 학습 → 비교 표 반환.

GUI Compare 대시보드 / CLI 벤치마크용 백엔드. validation split 기준으로
RMSE / R² / 학습시간을 집계.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.digital_twin.pipeline_compare import compare_models
    >>> # rng = np.random.default_rng(0)
    >>> # X = rng.standard_normal((30, 16))
    >>> # P = np.linspace(0,1,16).reshape(-1,1)
    >>> # table = compare_models(X, P, configs=[
    >>> #     ("pod", 3, "kriging"), ("pod", 5, "rbf")])
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _score(
    snapshots: NDArray[np.float64],
    params: NDArray[np.float64],
    reducer_kind: str,
    n_modes: int,
    surrogate_kind: str,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """단일 (reducer, n_modes, surrogate) 점수."""
    from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

    n = snapshots.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(2, int(round(n * val_ratio)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    t0 = time.perf_counter()
    pipe = NavierTwinPipeline(
        reducer_kind=reducer_kind,
        n_modes=int(max(1, min(n_modes, len(tr_idx) - 1))),
        surrogate_kind=surrogate_kind,
    )
    pipe.load_snapshots(snapshots[:, tr_idx], field_name="U")
    pipe.reduce()
    pipe.fit_surrogate(params[tr_idx])
    t_train = time.perf_counter() - t0

    c_true = pipe.state.reducer.encode(snapshots[:, val_idx])
    c_pred = pipe.state.surrogate.predict(params[val_idx])
    diff = c_true - c_pred
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((c_true - c_true.mean()) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot

    return {
        "reducer_kind": reducer_kind,
        "n_modes": int(n_modes),
        "surrogate_kind": surrogate_kind,
        "rmse": rmse,
        "r2": r2,
        "train_time_s": float(t_train),
    }


def compare_models(
    snapshots: NDArray[np.float64],
    params: NDArray[np.float64],
    configs: list[tuple[str, int, str]],
    *,
    val_ratio: float = 0.25,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """configs = [(reducer_kind, n_modes, surrogate_kind), ...] 각각 평가.

    실패한 조합은 rmse=inf, error 필드 기록. RMSE 오름차순 정렬.

    Returns:
        result 리스트 (각 row: reducer_kind/n_modes/surrogate_kind/rmse/r2/train_time_s).
    """
    rows: list[dict[str, Any]] = []
    for red, nm, sur in configs:
        try:
            rows.append(_score(snapshots, params, red, nm, sur, val_ratio, seed))
        except Exception as e:  # noqa: BLE001
            logger.warning("compare 실패 (%s,%d,%s): %s", red, nm, sur, e)
            rows.append({
                "reducer_kind": red,
                "n_modes": int(nm),
                "surrogate_kind": sur,
                "rmse": float("inf"),
                "r2": float("-inf"),
                "train_time_s": 0.0,
                "error": str(e),
            })
    rows.sort(key=lambda r: r["rmse"])
    logger.info("compare_models: %d configs, best rmse=%.6g", len(rows), rows[0]["rmse"])
    return rows


def rank_table(rows: list[dict[str, Any]]) -> str:
    """비교 결과를 텍스트 표로 포매팅."""
    header = f"{'rank':>4} {'reducer':>8} {'n_modes':>7} {'surrogate':>10} {'rmse':>10} {'r2':>8} {'time_s':>8}"
    lines = [header, "-" * len(header)]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i:>4} {r['reducer_kind']:>8} {r['n_modes']:>7} "
            f"{r['surrogate_kind']:>10} {r['rmse']:>10.4g} {r['r2']:>8.4g} "
            f"{r['train_time_s']:>8.3f}"
        )
    return "\n".join(lines)


__all__ = ["compare_models", "rank_table"]
