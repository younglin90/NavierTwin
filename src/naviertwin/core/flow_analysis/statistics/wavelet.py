"""웨이블릿 시간-주파수 분석 모듈.

CWT(Continuous Wavelet Transform) 로 비정상 유동의 과도(transient) 특징을
시간-주파수 평면에서 추출한다. PyWavelets 필요 (optional) —
없으면 FFT windowed(STFT) 경량 대안을 제공한다.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def continuous_wavelet(
    signal: NDArray[np.float64],
    dt: float = 1.0,
    scales: NDArray[np.float64] | None = None,
    wavelet: str = "morl",
) -> dict[str, NDArray[np.float64]]:
    """CWT 연속 웨이블릿 변환.

    Args:
        signal: 1D 시계열.
        dt: 샘플 간격 [s].
        scales: CWT 스케일 배열. None 이면 logspace 로 자동 선택.
        wavelet: PyWavelets 웨이블릿 이름.

    Returns:
        dict: {coefficients, frequencies, scales}

    Raises:
        RuntimeError: PyWavelets 미설치.
    """
    try:
        import pywt
    except ImportError as exc:
        raise RuntimeError(
            "pywt(PyWavelets) 설치 필요: pip install pywavelets"
        ) from exc

    signal = np.asarray(signal, dtype=np.float64).ravel()
    if scales is None:
        scales = np.logspace(0, np.log10(len(signal) / 4), 48)

    coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=dt)
    logger.debug("CWT: shape=%s, scales=%d", coeffs.shape, len(scales))
    return {
        "coefficients": np.asarray(coeffs),
        "frequencies": np.asarray(freqs),
        "scales": np.asarray(scales),
    }


def stft_fallback(
    signal: NDArray[np.float64],
    dt: float = 1.0,
    n_window: int = 64,
    n_overlap: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """PyWavelets 대신 사용할 수 있는 STFT 경량 구현.

    Returns:
        dict: {spectrogram (freq × time), frequencies, times}
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if n_overlap is None:
        n_overlap = n_window // 2
    step = n_window - n_overlap
    n_blocks = 1 + (len(signal) - n_window) // step
    win = np.hanning(n_window)
    windows = np.lib.stride_tricks.sliding_window_view(signal, n_window)[
        ::step
    ][:n_blocks]
    spec = np.fft.rfft(windows * win[np.newaxis, :], axis=1).T
    freqs = np.fft.rfftfreq(n_window, d=dt)
    times = np.arange(n_blocks) * step * dt
    return {
        "spectrogram": np.abs(spec) ** 2,
        "frequencies": freqs,
        "times": times,
    }


__all__ = ["continuous_wavelet", "stft_fallback"]
