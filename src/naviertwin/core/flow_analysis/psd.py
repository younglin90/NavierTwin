"""파워 스펙트럼 밀도 (Welch's method) — 시계열 주파수 분석.

CFD 시뮬레이션에서 추출한 시계열 데이터(예: 프로브 속도)의 주파수 성분을
Welch 평균화 주기도법으로 추정한다. 상용 후처리 툴 (Tecplot 360, EnSight,
MATLAB Signal Processing Toolbox)의 표준 PSD 알고리즘.

References:
    Welch, P., "The use of fast Fourier transform to estimate power
    spectra", IEEE Trans. Audio and Electroacoustics, 1967.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t = np.linspace(0, 10, 1000)
    >>> u = np.sin(2 * np.pi * 5 * t) + 0.1 * rng.standard_normal(1000)
    >>> from naviertwin.core.flow_analysis.psd import welch_psd
    >>> f, P = welch_psd(u, fs=100.0)
    >>> # 5Hz 근처에 피크 존재
    >>> int(f[P.argmax()])
    5
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _hann_window(n: int) -> NDArray[np.float64]:
    """Hann (Hanning) 창함수."""
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / max(n - 1, 1)))


def _hamming_window(n: int) -> NDArray[np.float64]:
    """Hamming 창함수."""
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / max(n - 1, 1))


def _boxcar_window(n: int) -> NDArray[np.float64]:
    """직사각 창함수 (창 미적용)."""
    return np.ones(n, dtype=np.float64)


_WINDOWS = {
    "hann": _hann_window,
    "hanning": _hann_window,
    "hamming": _hamming_window,
    "boxcar": _boxcar_window,
    "rect": _boxcar_window,
}


def welch_psd(
    x: NDArray[np.float64],
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str = "constant",
    scaling: str = "density",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Welch's method 기반 단변량 PSD 추정.

    Args:
        x: (N,) 시계열.
        fs: 샘플링 주파수 [Hz].
        nperseg: 세그먼트 길이. None이면 min(256, N).
        noverlap: 오버랩 샘플 수. None이면 nperseg // 2.
        window: 창함수 이름. "hann", "hamming", "boxcar".
        detrend: "constant" (평균 제거) 또는 "none".
        scaling: "density" (V²/Hz) 또는 "spectrum" (V²).

    Returns:
        Tuple[freq, P]: 양의 주파수 + PSD.

    Raises:
        ValueError: 매개변수 범위 오류.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")
    N = len(x)
    if N < 2:
        raise ValueError(f"x length must be >= 2, got {N}")

    if nperseg is None:
        nperseg = min(256, N)
    if nperseg > N:
        nperseg = N
    if nperseg <= 0:
        raise ValueError(f"nperseg must be > 0, got {nperseg}")

    if noverlap is None:
        noverlap = nperseg // 2
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError(
            f"noverlap must be in [0, nperseg-1], got {noverlap}"
        )

    if window not in _WINDOWS:
        raise ValueError(
            f"window '{window}' unknown; use one of {list(_WINDOWS)}"
        )
    win = _WINDOWS[window](nperseg)
    win_norm = (win * win).sum()  # window energy

    step = nperseg - noverlap
    n_seg = max(1, (N - noverlap) // step)

    starts = np.arange(n_seg) * step
    starts = starts[starts + nperseg <= N]
    frames = np.lib.stride_tricks.sliding_window_view(x, nperseg)[starts].copy()
    if detrend == "constant":
        frames -= frames.mean(axis=1, keepdims=True)
    S = np.fft.rfft(frames * win, axis=1)
    P_avg = np.mean(np.abs(S) ** 2, axis=0)

    if scaling == "density":
        P_avg = P_avg / (fs * win_norm)
    elif scaling == "spectrum":
        P_avg = P_avg / (win.sum() ** 2)
    else:
        raise ValueError(f"scaling '{scaling}' invalid; use 'density'/'spectrum'")

    # 1-sided 보정 (DC와 Nyquist 제외 ×2)
    P_avg = P_avg.copy()
    if scaling == "density":
        P_avg[1:-1] *= 2.0

    freq = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    logger.debug(
        "Welch PSD: N=%d, nperseg=%d, n_seg=%d, fs=%.4g", N, nperseg, n_seg, fs
    )
    return freq, P_avg


def cross_psd(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Welch's 방법 기반 크로스 스펙트럼 밀도 P_xy(f).

    Args:
        x, y: 1D 시계열 (같은 길이).
        fs: 샘플링 주파수.
        nperseg, noverlap, window: welch_psd와 동일.

    Returns:
        (freq, P_xy) — P_xy는 복소수.

    Raises:
        ValueError: x와 y 길이 불일치.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"x/y must be same-shape 1D, got {x.shape} vs {y.shape}")
    N = len(x)
    if nperseg is None:
        nperseg = min(256, N)
    if nperseg > N:
        nperseg = N
    if noverlap is None:
        noverlap = nperseg // 2
    win = _WINDOWS.get(window, _hann_window)(nperseg)
    win_norm = (win * win).sum()
    step = nperseg - noverlap
    n_seg = max(1, (N - noverlap) // step)

    starts = np.arange(n_seg) * step
    starts = starts[starts + nperseg <= N]
    x_frames = np.lib.stride_tricks.sliding_window_view(x, nperseg)[starts]
    y_frames = np.lib.stride_tricks.sliding_window_view(y, nperseg)[starts]
    sx = (x_frames - x_frames.mean(axis=1, keepdims=True)) * win
    sy = (y_frames - y_frames.mean(axis=1, keepdims=True)) * win
    Sx = np.fft.rfft(sx, axis=1)
    Sy = np.fft.rfft(sy, axis=1)
    P_xy = np.mean(np.conj(Sx) * Sy, axis=0) / (fs * win_norm)
    P_xy = P_xy.copy()
    P_xy[1:-1] *= 2.0
    freq = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freq, P_xy


def coherence(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Magnitude-squared coherence γ²(f) = |P_xy|² / (P_xx · P_yy) ∈ [0, 1].

    1.0이면 두 신호가 해당 주파수에서 선형 관계.

    Args:
        x, y: 1D 시계열.
        fs, nperseg, noverlap, window: welch_psd와 동일.

    Returns:
        (freq, coh) — coh ∈ [0, 1].
    """
    f, Pxx = welch_psd(x, fs, nperseg, noverlap, window)
    _, Pyy = welch_psd(y, fs, nperseg, noverlap, window)
    _, Pxy = cross_psd(x, y, fs, nperseg, noverlap, window)
    denom = np.maximum(Pxx * Pyy, 1e-30)
    coh = (np.abs(Pxy) ** 2) / denom
    return f, np.clip(coh, 0.0, 1.0)


__all__ = ["welch_psd", "cross_psd", "coherence"]
