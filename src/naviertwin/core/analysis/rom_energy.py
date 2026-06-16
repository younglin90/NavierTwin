"""POD / DMD / AE 축소 차수의 에너지 보존율 & 적정 차수 선택.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.rom_energy import energy_retention
    >>> sv = np.array([10., 5., 2., 0.5])
    >>> r = energy_retention(sv)
    >>> r[0] > 0 and r[-1] == 1.0
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def energy_retention(singular_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """에너지 누적 비율 (σ² 기준). 길이=n_modes."""
    sv = np.asarray(singular_values, dtype=np.float64).ravel()
    if sv.size == 0:
        raise ValueError("빈 singular_values")
    e = sv ** 2
    return np.cumsum(e) / (np.sum(e) + 1e-30)


def n_modes_for_energy(
    singular_values: NDArray[np.float64], threshold: float = 0.99,
) -> int:
    """주어진 에너지 임계값을 만족하는 최소 mode 수."""
    cum = energy_retention(singular_values)
    idx = np.searchsorted(cum, threshold)
    return int(min(idx + 1, cum.size))


def scree_elbow(singular_values: NDArray[np.float64]) -> int:
    """scree plot elbow (최대 2차 차분)."""
    sv = np.asarray(singular_values, dtype=np.float64).ravel()
    if sv.size < 3:
        return int(sv.size)
    d2 = np.diff(np.diff(sv))
    return int(np.argmax(d2) + 1)  # +1: diff 2회로 인덱스 이동


def energy_spectrum(singular_values: NDArray[np.float64]) -> dict[str, float]:
    """전체 / top-1 / top-3 / top-10 에너지 보존율."""
    sv = np.asarray(singular_values, dtype=np.float64).ravel()
    return _kernels.rom_energy_spectrum(sv)


__all__ = [
    "energy_retention", "n_modes_for_energy", "scree_elbow", "energy_spectrum",
]
