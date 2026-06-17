"""위상 잠금 평균 (Phase-Locked / Conditional Averaging).

회전 기계(터빈, 압축기, 펌프), 펄스 흐름, 와류 발산 등 주기적 신호에서
사이클의 같은 위상에서의 평균을 추출. 난류 변동을 제거하고 결정적
주기 성분만 남긴다.

상용 툴 대응:
    - Ansys CFX/Fluent: Time-Periodic / Phase Average
    - EnSight: Phase-locked sampling
    - PIV-DAQ 시스템: Phase Reference Synchronization

References:
    Reynolds, W.C. & Hussain, A.K.M.F., "The mechanics of an organized
    wave in turbulent shear flow", JFM, 1972.

Examples:
    >>> import numpy as np
    >>> # 가짜 주기 1.0의 사인파 + 노이즈
    >>> rng = np.random.default_rng(0)
    >>> t = np.linspace(0, 10, 1000)
    >>> u = np.sin(2 * np.pi * t / 1.0) + 0.3 * rng.standard_normal(1000)
    >>> from naviertwin.core.flow_analysis.phase_lock import phase_average
    >>> phases, mean, rms = phase_average(t, u, period=1.0, n_bins=20)
    >>> mean.shape
    (20,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def phase_average(
    t: NDArray[np.float64],
    signal: NDArray[np.float64],
    period: float,
    n_bins: int = 36,
    phase_offset: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """주기 신호를 위상 빈에 따라 평균/RMS 산출.

    Args:
        t: (N,) 시간 배열.
        signal: (N,) 또는 (N, ...) 신호 (시간이 첫 축).
        period: 사이클 주기.
        n_bins: 위상 빈 수.
        phase_offset: 위상 시작 오프셋 (라디안 단위가 아닌 사이클 분율).

    Returns:
        (phase_bin_centers [0, 2π], mean, rms).

    Raises:
        ValueError: 입력 형상 또는 매개변수 오류.
    """
    t = np.asarray(t, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    if t.shape[0] != signal.shape[0]:
        raise ValueError(
            f"t/signal length mismatch: {t.shape[0]} vs {signal.shape[0]}"
        )
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be > 0, got {n_bins}")

    # 위상 (사이클 분율 ∈ [0, 1))
    phase = ((t / period) - phase_offset) % 1.0
    # 빈 인덱스
    bin_idx = np.minimum((phase * n_bins).astype(int), n_bins - 1)

    sig_shape = signal.shape[1:]
    mean = np.zeros((n_bins,) + sig_shape, dtype=signal.dtype)
    rms = np.zeros((n_bins,) + sig_shape, dtype=signal.dtype)
    flat = signal.reshape(signal.shape[0], -1)
    sums = np.zeros((n_bins, flat.shape[1]), dtype=np.float64)
    sums_sq = np.zeros_like(sums)
    np.add.at(sums, bin_idx, flat)
    np.add.at(sums_sq, bin_idx, flat * flat)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    valid = counts > 0
    mean_flat = mean.reshape(n_bins, -1)
    rms_flat = rms.reshape(n_bins, -1)
    mean_flat[valid] = sums[valid] / counts[valid, np.newaxis]
    var_flat = np.zeros_like(sums)
    var_flat[valid] = sums_sq[valid] / counts[valid, np.newaxis] - mean_flat[valid] ** 2
    rms_flat[valid] = np.sqrt(np.maximum(var_flat[valid], 0.0))

    bin_centers = (np.arange(n_bins) + 0.5) / n_bins * 2.0 * np.pi
    logger.debug("phase_average: n_bins=%d, total samples=%d", n_bins, len(t))
    return bin_centers, mean, rms


def cycle_extract(
    t: NDArray[np.float64],
    signal: NDArray[np.float64],
    period: float,
    n_phase: int = 100,
    n_cycles: int | None = None,
    phase_offset: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """각 사이클을 추출해 (n_cycles, n_phase, ...) 배열로 반환.

    각 사이클은 동일 위상 격자(0~2π, n_phase 점)에 보간.

    Args:
        t: (N,) 시간.
        signal: (N, ...) 신호.
        period: 주기.
        n_phase: 위상당 샘플 수.
        n_cycles: 추출할 사이클 수. None이면 가용한 모두.
        phase_offset: 위상 시작 오프셋.

    Returns:
        (phase_grid, cycles): cycles 형상 (n_cycles, n_phase, ...).

    Raises:
        ValueError: 매개변수 오류.
    """
    t = np.asarray(t, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    if n_phase < 2:
        raise ValueError(f"n_phase must be >= 2, got {n_phase}")

    t_start = t[0] + phase_offset * period
    t_end = t[-1]
    n_full = int((t_end - t_start) // period)
    if n_full < 1:
        raise ValueError(
            f"signal too short to contain one full cycle (period={period})"
        )

    if n_cycles is None:
        n_cycles = n_full
    n_cycles = min(n_cycles, n_full)

    phase_grid = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)
    out_shape = (n_cycles, n_phase) + signal.shape[1:]
    cycle_starts = t_start + np.arange(n_cycles) * period
    t_query = cycle_starts[:, np.newaxis] + (phase_grid / (2.0 * np.pi)) * period
    idx = np.argmin(np.abs(t_query[:, :, np.newaxis] - t[np.newaxis, np.newaxis, :]), axis=2)
    cycles = signal[idx].reshape(out_shape)

    return phase_grid, cycles


def fundamental_period_from_acf(
    t: NDArray[np.float64],
    signal: NDArray[np.float64],
    min_period: float = 0.0,
    max_period: float | None = None,
) -> float:
    """자기상관 함수의 첫 번째 양의 피크에서 기본 주기 추정.

    Args:
        t: (N,) 시간 (균일 간격 가정).
        signal: (N,) 시계열.
        min_period: 최소 주기 (이보다 짧은 lag는 제외).
        max_period: 최대 주기 한계.

    Returns:
        추정 주기.

    Raises:
        ValueError: 시계열 너무 짧거나 균일 격자 아님.
    """
    t = np.asarray(t, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    n = len(t)
    if n < 10:
        raise ValueError(f"signal too short: {n}")
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError("non-uniform or non-monotonic t")

    sig0 = signal - signal.mean()
    var = np.dot(sig0, sig0)
    if var < 1e-30:
        raise ValueError("signal is constant")

    max_lag = n // 2 if max_period is None else int(max_period / dt)
    max_lag = min(max_lag, n - 1)
    min_lag = max(1, int(min_period / dt))

    corr = np.correlate(sig0, sig0, mode="full")
    acf = corr[n - 1:n + max_lag] / var

    # 첫 번째 로컬 최대값 (음수 → 양수 전환 후의 피크)
    # 단순: 첫 번째로 acf가 0 아래로 떨어졌다가 다시 올라가서 로컬 max 인 위치
    neg = np.flatnonzero(acf[min_lag:max_lag] < 0.0)
    if neg.size:
        start = min_lag + int(neg[0]) + 1
        if start < max_lag:
            center = acf[start:max_lag]
            local = np.flatnonzero((center > acf[start - 1:max_lag - 1]) & (center > acf[start + 1:max_lag + 1]))
            if local.size:
                return (start + int(local[0])) * dt
    # 폴백: 가장 큰 acf 위치 (lag>0)
    return int(np.argmax(acf[max(min_lag, 1):]) + max(min_lag, 1)) * dt


__all__ = [
    "phase_average",
    "cycle_extract",
    "fundamental_period_from_acf",
]
