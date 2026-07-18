"""조건부 샘플링 / 조건부 평균.

특정 트리거 이벤트(예: 임계 통과, 사분면 진입, 위상 정의)가 발생할 때마다
주변 시간 윈도우의 신호를 평균/표준편차로 추출. 이젝션 이벤트, 충격파
포착, 폭풍 시작 분석 등에 사용.

상용 툴 대응:
    - MATLAB: triggered averaging utilities
    - Tecplot 360: Time-Aware Trigger
    - 학술: Antonia 1981, "Conditional Sampling in Turbulence Measurement"

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> trigger = np.zeros(1000)
    >>> trigger[100] = trigger[300] = trigger[500] = 1.0
    >>> signal = rng.standard_normal((1000, 5))
    >>> from naviertwin.core.flow_analysis.conditional_sampling import (
    ...     trigger_average
    ... )
    >>> avg, count = trigger_average(signal, trigger, half_window=10)
    >>> avg.shape
    (21, 5)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")

logger = get_logger(__name__)


def find_threshold_crossings(
    signal: NDArray[np.float64],
    threshold: float,
    direction: str = "rising",
) -> NDArray[np.intp]:
    """임계값 교차 인덱스 반환.

    Args:
        signal: 1D 신호.
        threshold: 임계값.
        direction: "rising" (아래→위), "falling" (위→아래), "both".

    Returns:
        교차 인덱스 배열.

    Raises:
        ValueError: direction 오류.
    """
    s = np.asarray(signal, dtype=np.float64).ravel()
    above = s >= threshold
    transitions = np.diff(above.astype(int))
    if direction == "rising":
        idx = np.where(transitions == 1)[0] + 1
    elif direction == "falling":
        idx = np.where(transitions == -1)[0] + 1
    elif direction == "both":
        idx = np.where(transitions != 0)[0] + 1
    else:
        raise ValueError(
            f"direction must be 'rising'/'falling'/'both', got '{direction}'"
        )
    return idx.astype(np.intp)


def trigger_average(
    signal: NDArray[np.float64],
    trigger_indices: NDArray[np.intp] | NDArray[np.float64],
    half_window: int,
    return_std: bool = False,
) -> tuple[NDArray[np.float64], int] | tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """트리거 이벤트 주변 ±half_window의 평균 (앙상블 평균).

    Args:
        signal: (n_t,) 또는 (n_t, ...) 신호.
        trigger_indices: 트리거 발생 인덱스 (정수) 또는 trigger 신호 (≥1.0).
        half_window: 앞뒤 윈도우 반폭.
        return_std: True면 표준편차도 반환.

    Returns:
        (averaged, count): averaged 형상 (2*half+1, ...). 또는
        (averaged, std, count) if return_std=True.

    Raises:
        ValueError: half_window ≤ 0 또는 형상 오류.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if half_window <= 0:
        raise ValueError(f"half_window must be > 0, got {half_window}")
    n_t = signal.shape[0]
    win_len = 2 * half_window + 1
    out_shape = (win_len,) + signal.shape[1:]

    trig = np.asarray(trigger_indices)
    if trig.dtype.kind == "f":
        idx = np.where(trig >= 1.0)[0]
    else:
        idx = trig.astype(np.intp)

    valid = idx[(idx - half_window >= 0) & (idx + half_window < n_t)]
    if len(valid) == 0:
        if return_std:
            return np.zeros(out_shape), np.zeros(out_shape), 0
        return np.zeros(out_shape), 0

    return _kernels.trigger_average_accum(
        signal,
        valid.astype(np.int64),
        int(half_window),
        bool(return_std),
    )


def conditional_average(
    signal: NDArray[np.float64],
    condition: NDArray[np.bool_] | NDArray[np.float64] | Callable[[NDArray[np.float64]], NDArray[np.bool_]],
) -> tuple[NDArray[np.float64], int]:
    """조건이 참인 시점들의 평균.

    Args:
        signal: (n_t, ...) 신호.
        condition: (n_t,) 부울/정수 마스크 또는 signal 첫 차원에 적용 가능한 함수.

    Returns:
        (averaged_field, n_samples).

    Raises:
        ValueError: 형상 불일치.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if callable(condition):
        cond_arr = np.asarray(condition(signal), dtype=bool)
    else:
        cond_arr = np.asarray(condition, dtype=bool)

    if cond_arr.shape != (signal.shape[0],):
        raise ValueError(
            f"condition shape {cond_arr.shape} != ({signal.shape[0]},)"
        )

    sub = signal[cond_arr]
    if len(sub) == 0:
        return np.zeros(signal.shape[1:], dtype=signal.dtype), 0
    return sub.mean(axis=0), int(len(sub))


def quadrant_masks(
    up: NDArray[np.float64],
    vp: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """u'v' 평면의 4사분면 부울 마스크.

    Q1: u'>0, v'>0 (외향 가속)
    Q2: u'<0, v'>0 (이젝션)
    Q3: u'<0, v'<0 (내향 감속)
    Q4: u'>0, v'<0 (스윕)

    Args:
        up, vp: 변동 신호.

    Returns:
        (Q1_mask, Q2_mask, Q3_mask, Q4_mask).

    Raises:
        ValueError: 형상 불일치.
    """
    up = np.asarray(up, dtype=np.float64).ravel()
    vp = np.asarray(vp, dtype=np.float64).ravel()
    if up.shape != vp.shape:
        raise ValueError(f"up/vp shape mismatch: {up.shape} vs {vp.shape}")
    Q1 = (up > 0) & (vp > 0)
    Q2 = (up < 0) & (vp > 0)
    Q3 = (up < 0) & (vp < 0)
    Q4 = (up > 0) & (vp < 0)
    return Q1, Q2, Q3, Q4


def event_duration_stats(
    condition: NDArray[np.bool_],
    dt: float = 1.0,
) -> dict[str, float | int]:
    """조건이 참인 연속 구간의 지속 시간 통계.

    Args:
        condition: (N,) 부울 시퀀스.
        dt: 시간 간격.

    Returns:
        dict with keys: n_events, mean_duration, max_duration,
        total_active_time, active_fraction.
    """
    cond = np.asarray(condition, dtype=bool).ravel()
    if len(cond) == 0:
        return {
            "n_events": 0, "mean_duration": 0.0, "max_duration": 0.0,
            "total_active_time": 0.0, "active_fraction": 0.0,
        }

    # 연속 True 그룹의 시작/끝
    diff = np.diff(np.concatenate([[False], cond, [False]]).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    durations = (ends - starts) * dt

    n_events = len(starts)
    if n_events == 0:
        return {
            "n_events": 0, "mean_duration": 0.0, "max_duration": 0.0,
            "total_active_time": 0.0, "active_fraction": 0.0,
        }

    return {
        "n_events": int(n_events),
        "mean_duration": float(durations.mean()),
        "max_duration": float(durations.max()),
        "total_active_time": float(durations.sum()),
        "active_fraction": float(cond.sum()) / len(cond),
    }


__all__ = [
    "find_threshold_crossings",
    "trigger_average",
    "conditional_average",
    "quadrant_masks",
    "event_duration_stats",
]
