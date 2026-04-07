"""FFT 및 PSD 주파수 분석 모듈.

1D 시계열 신호 및 공간 필드 스냅샷에 대한
FFT, Welch PSD, 지배 주파수 피크 탐지 함수를 제공한다.

Examples:
    기본 FFT::

        import numpy as np
        from naviertwin.core.flow_analysis.statistics.fft_psd import (
            compute_fft,
            compute_psd,
            find_dominant_frequencies,
        )

        dt = 0.01
        t = np.arange(0, 10, dt)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)

        freqs, amps = compute_fft(signal, dt)
        peaks = find_dominant_frequencies(freqs, amps, n_peaks=3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def compute_fft(
    signal: NDArray[np.float64],
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """1D 시계열 신호의 FFT 를 계산한다.

    단측 스펙트럼(양수 주파수)만 반환한다.
    진폭은 단측 스펙트럼 정규화(× 2 / N)를 적용한다.

    Args:
        signal: 시계열 데이터, shape ``(N,)`` [임의 단위].
        dt: 샘플링 간격 [s]. 0 보다 커야 한다.

    Returns:
        ``(freqs, amplitudes)`` 튜플:
            - ``freqs``: 주파수 배열 [Hz], shape ``(N//2,)``.
            - ``amplitudes``: 진폭 배열 [signal 단위], shape ``(N//2,)``.

    Raises:
        ValueError: ``dt`` 가 0 이하인 경우 또는 signal 이 비어있는 경우.

    Examples:
        >>> dt = 0.01
        >>> t = np.arange(0, 1, dt)
        >>> sig = np.sin(2 * np.pi * 10 * t)
        >>> freqs, amps = compute_fft(sig, dt)
    """
    if dt <= 0:
        raise ValueError(f"dt 는 0 보다 커야 합니다. 입력값: {dt}")
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        raise ValueError("signal 이 비어있습니다.")

    n = len(signal)
    fft_raw = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=dt)

    # 단측 정규화 진폭: 2/N × |X[k]| (DC 와 Nyquist 제외)
    amps = np.abs(fft_raw) * 2.0 / n
    amps[0] /= 2.0  # DC 성분
    if n % 2 == 0:
        amps[-1] /= 2.0  # Nyquist 성분

    logger.debug(
        "compute_fft: N=%d, dt=%.4f, 주파수 해상도=%.4f Hz",
        n,
        dt,
        freqs[1] if len(freqs) > 1 else 0.0,
    )
    return freqs, amps


def compute_psd(
    signal: NDArray[np.float64],
    dt: float,
    window: str = "hann",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Welch 방법으로 Power Spectral Density(PSD) 를 계산한다.

    ``scipy.signal.welch`` 를 사용하며, scipy 가 없으면 단순 periodogram 으로
    폴백한다.

    Args:
        signal: 시계열 데이터, shape ``(N,)`` [임의 단위].
        dt: 샘플링 간격 [s].
        window: 윈도우 함수 이름. 기본값 ``"hann"``.
            scipy 가 없으면 무시된다.

    Returns:
        ``(freqs, psd)`` 튜플:
            - ``freqs``: 주파수 배열 [Hz].
            - ``psd``: PSD 배열 [단위²/Hz].

    Raises:
        ValueError: ``dt`` 가 0 이하인 경우.
    """
    if dt <= 0:
        raise ValueError(f"dt 는 0 보다 커야 합니다. 입력값: {dt}")
    signal = np.asarray(signal, dtype=np.float64)

    fs = 1.0 / dt

    try:
        from scipy import signal as sp_signal

        freqs, psd = sp_signal.welch(
            signal,
            fs=fs,
            window=window,
            nperseg=min(256, len(signal)),
            scaling="density",
        )
        logger.debug("compute_psd: Welch 방법 (scipy) 사용")
    except ImportError:
        logger.warning(
            "scipy 가 없습니다. 단순 periodogram 으로 폴백합니다."
        )
        # 단순 periodogram (scipy 없음)
        n = len(signal)
        fft_raw = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=dt)
        psd = (np.abs(fft_raw) ** 2) / (fs * n)
        psd[1:-1] *= 2.0  # 단측 스펙트럼 보정

    return freqs, psd


def find_dominant_frequencies(
    freqs: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    n_peaks: int = 5,
) -> list[dict[str, float]]:
    """진폭 스펙트럼에서 지배 주파수 피크를 탐색한다.

    DC(f=0) 성분을 제외하고 진폭이 큰 순서로 ``n_peaks`` 개의 피크를
    반환한다. scipy.signal.find_peaks 가 가능하면 실제 피크를, 없으면
    단순 정렬로 선택한다.

    Args:
        freqs: 주파수 배열 [Hz], shape ``(N,)``.
        amplitudes: 진폭 배열, shape ``(N,)``.
        n_peaks: 반환할 최대 피크 개수.

    Returns:
        주파수 피크 딕셔너리 리스트:
            ``[{"frequency": f, "amplitude": a, "strouhal": 0.0}, ...]``

        Strouhal 수는 0.0 으로 초기화된다 (특성 길이/속도 미입력).

    Examples:
        >>> peaks = find_dominant_frequencies(freqs, amps, n_peaks=3)
        >>> print(peaks[0]["frequency"])
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)

    # DC 제외
    valid_mask = freqs > 0.0
    valid_freqs = freqs[valid_mask]
    valid_amps = amplitudes[valid_mask]

    if len(valid_freqs) == 0:
        return []

    peak_indices: NDArray[np.intp]
    try:
        from scipy.signal import find_peaks

        peak_indices, _ = find_peaks(valid_amps, height=0.0)
        # 진폭 기준 내림차순
        peak_indices = peak_indices[
            np.argsort(valid_amps[peak_indices])[::-1]
        ]
    except ImportError:
        # scipy 없으면 단순 내림차순 정렬
        peak_indices = np.argsort(valid_amps)[::-1]

    peak_indices = peak_indices[:n_peaks]

    result: list[dict[str, float]] = []
    for idx in peak_indices:
        result.append(
            {
                "frequency": float(valid_freqs[idx]),
                "amplitude": float(valid_amps[idx]),
                "strouhal": 0.0,  # 특성 스케일 미지정
            }
        )

    logger.debug(
        "find_dominant_frequencies: %d 피크 탐지", len(result)
    )
    return result


def compute_field_fft(
    snapshots: NDArray[np.float64],
    dt: float,
    point_idx: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """공간 필드 스냅샷 배열에서 FFT 를 계산한다.

    특정 점의 시계열 또는 전체 공간 평균 시계열에 대해 FFT 를 수행한다.

    Args:
        snapshots: 스냅샷 배열.
            - ``(N_time, N_points)``: 스칼라 필드
            - ``(N_time, N_points, N_comp)``: 벡터 필드 (크기로 처리)
        dt: 타임스텝 간격 [s].
        point_idx: 분석할 점 인덱스.
            ``None`` 이면 공간 평균 시계열을 사용한다.

    Returns:
        ``(freqs, amplitudes)`` 튜플 (:func:`compute_fft` 와 동일 형식).

    Raises:
        ValueError: ``snapshots`` 차원이 올바르지 않은 경우.
        IndexError: ``point_idx`` 가 범위를 벗어난 경우.

    Examples:
        >>> snaps = np.random.rand(100, 500)  # 100 타임스텝, 500 점
        >>> freqs, amps = compute_field_fft(snaps, dt=0.01)
    """
    snapshots = np.asarray(snapshots, dtype=np.float64)

    if snapshots.ndim == 1:
        raise ValueError(
            "snapshots 는 2D 또는 3D 배열이어야 합니다. "
            f"입력 shape: {snapshots.shape}"
        )

    # 벡터 필드면 크기(norm)로 환산
    if snapshots.ndim == 3:
        snapshots = np.linalg.norm(snapshots, axis=-1)  # (N_time, N_points)

    n_time, n_points = snapshots.shape

    if point_idx is not None:
        if not (0 <= point_idx < n_points):
            raise IndexError(
                f"point_idx={point_idx} 가 범위를 벗어납니다 "
                f"(0 ~ {n_points - 1})."
            )
        signal = snapshots[:, point_idx]
        logger.debug(
            "compute_field_fft: 점 %d 의 시계열 FFT", point_idx
        )
    else:
        signal = snapshots.mean(axis=1)
        logger.debug("compute_field_fft: 공간 평균 FFT")

    return compute_fft(signal, dt)
