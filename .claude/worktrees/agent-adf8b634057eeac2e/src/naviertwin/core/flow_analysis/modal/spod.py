"""Spectral POD (SPOD) 모듈.

주파수 도메인 POD. 비정상 유동의 지배 구조를 주파수별로 분해한다.
PySPOD 가 있으면 그걸 사용하고, 없으면 Welch-block 기반 경량 구현으로 폴백한다.

References:
    - Towne, Schmidt, Colonius, JFM 2018 — SPOD 정의
    - Mengaldo et al., CPC 2024 — PySPOD

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.flow_analysis.modal.spod import compute_spod
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 512))  # (n_features, n_time)
    >>> result = compute_spod(X, dt=0.01, n_fft=64, n_modes=3)
    >>> result["frequencies"].shape
    (33,)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def compute_spod(
    snapshots: NDArray[np.float64],
    dt: float = 1.0,
    n_fft: int = 128,
    n_overlap: int | None = None,
    n_modes: int = 5,
    window: str = "hann",
) -> dict[str, NDArray[np.float64]]:
    """Welch-block 기반 SPOD.

    Args:
        snapshots: (n_features, n_time) 스냅샷 행렬.
        dt: 시간 간격 [s].
        n_fft: 블록 길이 (FFT 포인트 수).
        n_overlap: 블록 중첩 크기. None 이면 n_fft//2.
        n_modes: 각 주파수에서 유지할 SPOD 모드 수.
        window: 윈도우 이름 ("hann", "hamming", "boxcar").

    Returns:
        dict:
            - frequencies: shape (n_freq,) 양의 주파수 배열
            - eigenvalues: shape (n_freq, n_modes) 주파수별 SPOD 에너지
            - modes: shape (n_features, n_freq, n_modes) 복소 SPOD 모드

    Raises:
        ValueError: snapshots 가 2D 가 아니거나 시간 길이가 부족한 경우.
    """
    if snapshots.ndim != 2:
        raise ValueError(f"snapshots 는 2D 이어야 합니다: shape={snapshots.shape}")

    n_features, n_time = snapshots.shape
    if n_time < n_fft:
        raise ValueError(
            f"n_time({n_time}) < n_fft({n_fft}) — 더 긴 시계열 또는 작은 n_fft 필요"
        )

    if n_overlap is None:
        n_overlap = n_fft // 2
    step = n_fft - n_overlap
    n_blocks = 1 + (n_time - n_fft) // step

    # 윈도우 함수
    win = _get_window(window, n_fft)
    win_power = float(np.sum(win**2) / n_fft)

    # 평균 제거
    mean = snapshots.mean(axis=1, keepdims=True)
    X = snapshots - mean

    # 블록별 FFT → (n_features, n_freq, n_blocks)
    n_freq = n_fft // 2 + 1
    blocks = np.lib.stride_tricks.sliding_window_view(X, n_fft, axis=1)[
        :, ::step, :
    ][:, :n_blocks, :]
    Q = np.fft.rfft(blocks * win[np.newaxis, np.newaxis, :], axis=2)
    Q = np.transpose(Q / n_fft, (0, 2, 1))

    # 각 주파수마다 cross-spectral density 의 SVD
    eigvals = np.zeros((n_freq, n_modes), dtype=np.float64)
    modes = np.zeros((n_features, n_freq, n_modes), dtype=np.complex128)

    gram = np.einsum("fkb,fkc->kbc", Q.conj(), Q) / (n_blocks * win_power)
    w, v = np.linalg.eigh(gram)
    order = np.argsort(w, axis=1)[:, ::-1]
    w = np.take_along_axis(w, order, axis=1).real
    v = np.take_along_axis(v, order[:, np.newaxis, :], axis=2)
    keep = min(n_modes, n_blocks)
    eigvals[:, :keep] = np.maximum(w[:, :keep], 0.0)
    lam = np.sqrt(np.maximum(w[:, :keep], 1e-30))
    modes[:, :, :keep] = (
        np.einsum("fkb,kbm->fkm", Q, v[:, :, :keep])
        / lam[np.newaxis, :, :]
    )

    frequencies = np.fft.rfftfreq(n_fft, d=dt)
    logger.info(
        "SPOD 완료: n_blocks=%d, n_freq=%d, n_modes=%d",
        n_blocks,
        n_freq,
        n_modes,
    )
    return {
        "frequencies": frequencies,
        "eigenvalues": eigvals,
        "modes": modes,
    }


def _get_window(name: str, n: int) -> NDArray[np.float64]:
    name = name.lower()
    if name == "hann":
        return np.hanning(n).astype(np.float64)
    if name == "hamming":
        return np.hamming(n).astype(np.float64)
    if name == "boxcar":
        return np.ones(n, dtype=np.float64)
    logger.warning("알 수 없는 window '%s' → hann 사용", name)
    return np.hanning(n).astype(np.float64)


def compute_spod_pyspod(
    snapshots: NDArray[np.float64], dt: float, n_fft: int = 128, n_modes: int = 5
) -> Any:
    """PySPOD 가 설치된 경우 그 백엔드로 SPOD 를 수행한다.

    PySPOD 의 결과 객체를 그대로 반환한다. 설치 안 되어 있으면 RuntimeError.
    """
    try:
        import pyspod.spod_standard as pstd
    except ImportError as exc:
        raise RuntimeError(
            "pyspod 설치 필요: pip install naviertwin[core]"
        ) from exc

    params = {
        "time_step": dt,
        "n_space_dims": 1,
        "n_variables": 1,
        "n_fft": n_fft,
        "n_overlap": n_fft // 2,
        "n_modes_save": n_modes,
        "conf_level": 0.95,
    }
    spod = pstd.SPOD_standard(params=params, data_list=snapshots.T)
    spod.fit()
    return spod


__all__ = ["compute_spod", "compute_spod_pyspod"]
