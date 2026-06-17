"""POD 모드 검사 — 모드별 공간 패턴/시간 시리즈/에너지 통계 패키징.

학습된 POD 모델에서 사용자 친화적 dict로 각 모드의 정보를 추출.
GUI 시각화, Jupyter notebook 분석에 사용.

Examples:
    >>> import numpy as np
    >>> from numpy.linalg import svd as _svd
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((40, 30))
    >>> U, s, Vt = _svd(X, full_matrices=False)
    >>> from naviertwin.core.dimensionality_reduction.mode_inspection import (
    ...     mode_summary
    ... )
    >>> info = mode_summary(U[:, :5], s[:5], Vt[:5, :])
    >>> "spatial" in info[0]
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def mode_summary(
    spatial_modes: NDArray[np.float64],
    singular_values: NDArray[np.float64],
    temporal_modes: NDArray[np.float64] | None = None,
) -> list[dict]:
    """모드별 패키지된 dict 리스트 반환.

    Args:
        spatial_modes: (n_x, r) 공간 모드.
        singular_values: (r,) 특이값.
        temporal_modes: (r, n_t) 시간 모드 (선택, V 행렬).

    Returns:
        list of {"index", "spatial", "singular_value", "energy_fraction",
                 "spatial_norm", "temporal", "temporal_freq_dom"}.

    Raises:
        ValueError: 형상 불일치.
    """
    U = np.asarray(spatial_modes, dtype=np.float64)
    s = np.asarray(singular_values, dtype=np.float64).ravel()
    if U.ndim != 2 or U.shape[1] != s.shape[0]:
        raise ValueError(
            f"spatial_modes/singular_values mismatch: "
            f"{U.shape}, {s.shape}"
        )

    total_energy = float(np.sum(s ** 2)) + 1e-30

    Vt: NDArray[np.float64] | None = None
    if temporal_modes is not None:
        Vt = np.asarray(temporal_modes, dtype=np.float64)
        if Vt.ndim != 2 or Vt.shape[0] != s.shape[0]:
            raise ValueError(
                f"temporal_modes shape {Vt.shape} != ({s.shape[0]}, n_t)"
            )

    out = []
    k = 0
    while k < s.shape[0]:
        entry: dict = {
            "index": k,
            "spatial": U[:, k].copy(),
            "singular_value": float(s[k]),
            "energy_fraction": float(s[k] ** 2 / total_energy),
            "spatial_norm": float(np.linalg.norm(U[:, k])),
        }
        if Vt is not None:
            entry["temporal"] = Vt[k, :].copy()
            entry["temporal_freq_dom"] = _dominant_frequency(Vt[k, :])
        out.append(entry)
        k += 1
    return out


def _dominant_frequency(signal: NDArray[np.float64]) -> float:
    """FFT 피크의 정규화 주파수 (∈ [0, 0.5]) — 시간 모드 진동 식별."""
    n = len(signal)
    if n < 4:
        return 0.0
    s = signal - signal.mean()
    spec = np.abs(np.fft.rfft(s))
    if len(spec) <= 1:
        return 0.0
    # DC 제외
    peak_idx = int(np.argmax(spec[1:]) + 1)
    return float(peak_idx / n)


def mode_orthogonality(
    spatial_modes: NDArray[np.float64],
) -> dict[str, float]:
    """공간 모드 직교성 진단.

    Args:
        spatial_modes: (n_x, r).

    Returns:
        dict {max_off_diag, frobenius_dev, max_diag_dev}.

    Raises:
        ValueError: 2D 아님.
    """
    U = np.asarray(spatial_modes, dtype=np.float64)
    if U.ndim != 2:
        raise ValueError(f"spatial_modes must be 2D, got {U.shape}")
    G = U.T @ U
    r = G.shape[0]
    diag = np.diag(G)
    off = G - np.diag(diag)
    return {
        "max_off_diag": float(np.abs(off).max()) if r > 1 else 0.0,
        "frobenius_dev": float(np.linalg.norm(G - np.eye(r))),
        "max_diag_dev": float(np.abs(diag - 1.0).max()),
    }


def project_snapshot(
    snapshot: NDArray[np.float64],
    spatial_modes: NDArray[np.float64],
    mean: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """스냅샷을 모드 공간에 투영.

    Args:
        snapshot: (n_x,) 또는 (n_x, n_samples).
        spatial_modes: (n_x, r) 직교 정규.
        mean: (n_x,) 평균. None이면 0.

    Returns:
        (r,) 또는 (r, n_samples) 계수.

    Raises:
        ValueError: 차원 불일치.
    """
    U = np.asarray(spatial_modes, dtype=np.float64)
    s = np.asarray(snapshot, dtype=np.float64)
    if U.ndim != 2:
        raise ValueError(f"spatial_modes must be 2D, got {U.shape}")
    n_x = U.shape[0]
    if s.ndim == 1:
        if s.shape[0] != n_x:
            raise ValueError(
                f"snapshot length {s.shape[0]} != n_x {n_x}"
            )
        if mean is not None:
            s = s - np.asarray(mean, dtype=np.float64)
        return U.T @ s
    if s.ndim != 2:
        raise ValueError(f"snapshot must be 1D or 2D, got {s.shape}")
    if s.shape[0] != n_x:
        raise ValueError(
            f"snapshot rows {s.shape[0]} != n_x {n_x}"
        )
    if mean is not None:
        s = s - np.asarray(mean, dtype=np.float64)[:, None]
    return U.T @ s


def time_coefficient_statistics(
    coeffs: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """시간 계수의 통계량 dict.

    Args:
        coeffs: (n_t, r) 또는 (r, n_t).

    Returns:
        dict {mean, std, range, peak_freq} 모두 (r,).
    """
    C = np.asarray(coeffs, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError(f"coeffs must be 2D, got {C.shape}")
    # 더 큰 차원이 시간이라고 가정
    if C.shape[1] > C.shape[0]:
        C = C.T  # → (n_t, r)
    out = {
        "mean": C.mean(axis=0),
        "std": C.std(axis=0),
        "range": C.max(axis=0) - C.min(axis=0),
        "peak_freq": np.apply_along_axis(_dominant_frequency, 0, C),
    }
    return out


__all__ = [
    "mode_summary",
    "mode_orthogonality",
    "project_snapshot",
    "time_coefficient_statistics",
]
