"""스냅샷 시간 보간 — 임의 시각 t에서의 장(field) 보간.

CFD 결과는 이산 시각 t_0, t_1, ... 에서만 저장되지만, 사용자는
임의의 시간(예: 0.5 dt 위치, 또는 등간격 재샘플링)에서 장이 필요하다.
선형/3차 큐빅 스플라인/Hermite 보간을 제공한다.

상용 툴 대응:
    - Tecplot 360: Time-Aware → Smooth Time Sliding
    - Ansys CFD-Post: Animation interpolation
    - EnSight: Time interpolation between time steps

Examples:
    >>> import numpy as np
    >>> snaps = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (n_t, n_space)
    >>> times = np.array([0.0, 1.0, 2.0])
    >>> from naviertwin.core.flow_analysis.time_interp import interp_field
    >>> u_at_05 = interp_field(snaps, times, t_query=0.5)
    >>> u_at_05.tolist()
    [2.0, 3.0]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def interp_field(
    snapshots: NDArray[np.float64],
    times: NDArray[np.float64],
    t_query: float | NDArray[np.float64],
    method: str = "linear",
) -> NDArray[np.float64]:
    """임의 시각에서의 필드 보간.

    Args:
        snapshots: (n_t, ...) 시간-순 스냅샷.
        times: (n_t,) 단조 증가 시간 배열.
        t_query: 보간할 시각 (스칼라 또는 배열).
        method: "linear", "cubic" (3차 자연 스플라인), "nearest".

    Returns:
        보간된 필드 — t_query가 스칼라면 (...), 배열이면 (n_q, ...).

    Raises:
        ValueError: 형상 불일치 또는 method 오류.
    """
    snapshots = np.asarray(snapshots, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if snapshots.shape[0] != times.shape[0]:
        raise ValueError(
            f"snapshots/times mismatch: {snapshots.shape[0]} vs {times.shape[0]}"
        )
    if times.ndim != 1:
        raise ValueError(f"times must be 1D, got {times.shape}")
    if not np.all(np.diff(times) > 0):
        raise ValueError("times must be strictly increasing")

    t_query_arr = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
    scalar_input = np.isscalar(t_query) or (
        isinstance(t_query, np.ndarray) and t_query.ndim == 0
    )

    if method == "linear":
        out = _linear_interp(snapshots, times, t_query_arr)
    elif method == "cubic":
        out = _cubic_interp(snapshots, times, t_query_arr)
    elif method == "nearest":
        out = _nearest_interp(snapshots, times, t_query_arr)
    else:
        raise ValueError(
            f"method '{method}' invalid; use 'linear'/'cubic'/'nearest'"
        )

    return out[0] if scalar_input else out


def _linear_interp(
    snaps: NDArray[np.float64],
    times: NDArray[np.float64],
    tq: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_t = times.shape[0]
    out_shape = (tq.shape[0],) + snaps.shape[1:]
    out = np.zeros(out_shape, dtype=snaps.dtype)
    for k, t in enumerate(tq):
        idx = np.searchsorted(times, t)
        if idx == 0:
            out[k] = snaps[0]
        elif idx >= n_t:
            out[k] = snaps[-1]
        else:
            t0, t1 = times[idx - 1], times[idx]
            f0, f1 = snaps[idx - 1], snaps[idx]
            alpha = (t - t0) / (t1 - t0)
            out[k] = (1 - alpha) * f0 + alpha * f1
    return out


def _nearest_interp(
    snaps: NDArray[np.float64],
    times: NDArray[np.float64],
    tq: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_t = times.shape[0]
    out_shape = (tq.shape[0],) + snaps.shape[1:]
    out = np.zeros(out_shape, dtype=snaps.dtype)
    for k, t in enumerate(tq):
        idx = np.searchsorted(times, t)
        if idx == 0:
            out[k] = snaps[0]
        elif idx >= n_t:
            out[k] = snaps[-1]
        else:
            if abs(t - times[idx - 1]) < abs(t - times[idx]):
                out[k] = snaps[idx - 1]
            else:
                out[k] = snaps[idx]
    return out


def _cubic_interp(
    snaps: NDArray[np.float64],
    times: NDArray[np.float64],
    tq: NDArray[np.float64],
) -> NDArray[np.float64]:
    """자연 3차 스플라인 보간 (각 공간 위치 별로 1D 스플라인)."""
    n_t = times.shape[0]
    if n_t < 4:
        # 데이터 부족 → 선형으로 폴백
        logger.warning("cubic 보간 데이터 부족 (n_t=%d), linear로 폴백", n_t)
        return _linear_interp(snaps, times, tq)

    flat = snaps.reshape(n_t, -1)
    out_flat = np.zeros((tq.shape[0], flat.shape[1]), dtype=snaps.dtype)
    for j in range(flat.shape[1]):
        out_flat[:, j] = _natural_cubic_spline(times, flat[:, j], tq)
    return out_flat.reshape((tq.shape[0],) + snaps.shape[1:])


def _natural_cubic_spline(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    xq: NDArray[np.float64],
) -> NDArray[np.float64]:
    """자연 3차 스플라인 (양 끝 2차 도함수 = 0)."""
    n = len(x) - 1
    h = np.diff(x)

    # tridiagonal system for second derivatives
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    A[0, 0] = 1.0
    A[n, n] = 1.0
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    M = np.linalg.solve(A, b)

    out = np.zeros_like(xq)
    for k, q in enumerate(xq):
        idx = np.searchsorted(x, q)
        if idx == 0:
            out[k] = y[0]
            continue
        if idx > n:
            out[k] = y[n]
            continue
        i = idx - 1
        dx = q - x[i]
        hi = h[i]
        a = (M[i + 1] - M[i]) / (6.0 * hi)
        bi = M[i] / 2.0
        ci = (y[i + 1] - y[i]) / hi - hi * (M[i + 1] + 2 * M[i]) / 6.0
        out[k] = y[i] + ci * dx + bi * dx ** 2 + a * dx ** 3
    return out


def resample_uniform(
    snapshots: NDArray[np.float64],
    times: NDArray[np.float64],
    n_uniform: int,
    method: str = "linear",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """비균일 시간 시리즈를 균일 간격으로 재샘플링.

    Args:
        snapshots: (n_t, ...) 입력 스냅샷.
        times: (n_t,) 비균일 시간.
        n_uniform: 출력 샘플 수.
        method: 보간법.

    Returns:
        (uniform_times, uniform_snapshots).
    """
    times = np.asarray(times, dtype=np.float64)
    t_uniform = np.linspace(times[0], times[-1], n_uniform)
    out = interp_field(snapshots, times, t_uniform, method=method)
    return t_uniform, out


def time_average_window(
    snapshots: NDArray[np.float64],
    times: NDArray[np.float64],
    t_center: float,
    half_width: float,
) -> NDArray[np.float64]:
    """슬라이딩 윈도우 시간 평균 ⟨u⟩(t_c, Δt/2).

    Args:
        snapshots: (n_t, ...) 시계열.
        times: (n_t,) 시간.
        t_center: 윈도우 중심.
        half_width: 윈도우 반폭.

    Returns:
        평균 필드 (윈도우 내 스냅샷 평균). 빈 윈도우는 0 반환.

    Raises:
        ValueError: half_width ≤ 0.
    """
    if half_width <= 0:
        raise ValueError(f"half_width must be > 0, got {half_width}")
    times = np.asarray(times, dtype=np.float64)
    snapshots = np.asarray(snapshots, dtype=np.float64)

    mask = (times >= t_center - half_width) & (times <= t_center + half_width)
    if not mask.any():
        return np.zeros(snapshots.shape[1:], dtype=snapshots.dtype)
    return snapshots[mask].mean(axis=0)


__all__ = [
    "interp_field",
    "resample_uniform",
    "time_average_window",
]
