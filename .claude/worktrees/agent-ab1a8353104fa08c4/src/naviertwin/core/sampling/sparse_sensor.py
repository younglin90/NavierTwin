"""스파스 센서 배치 최적화 — QR 피벗 기반 최적 위치 선택.

QR 분해의 피벗 열을 이용해 POD/SVD 기저 행렬에서 최적 센서 위치를 선택한다.
선택된 센서로 상태를 재구성하는 최소자승 기반 복원 함수도 제공한다.

References:
    Manohar et al., "Data-Driven Sparse Sensor Placement and Reconstruction:
    Demonstrating the Benefits of Exploiting Known Patterns", IEEE Control
    Systems Magazine, 2018.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> U = rng.standard_normal((100, 10))  # 100 공간, 10 모드
    >>> from naviertwin.core.sampling.sparse_sensor import select_sensors
    >>> sensors = select_sensors(U, n_sensors=5)
    >>> sensors.shape
    (5,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def select_sensors(
    basis: NDArray[np.float64],
    n_sensors: int,
    method: str = "qr",
) -> NDArray[np.intp]:
    """POD/SVD 기저로부터 최적 센서 위치를 선택한다.

    Args:
        basis: (n_space, n_modes) — 공간 기저 행렬 (POD 모드 등).
        n_sensors: 선택할 센서 수. n_modes 이상 이어야 재구성이 가능하다.
        method: 알고리즘. 현재 ``"qr"`` (QR-피벗)만 지원.

    Returns:
        (n_sensors,) 정수 인덱스 배열 — 선택된 공간 위치.

    Raises:
        ValueError: basis가 2D가 아니거나 n_sensors 범위가 잘못된 경우.
        NotImplementedError: 지원하지 않는 method인 경우.
    """
    basis = np.asarray(basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"basis must be 2D, got shape {basis.shape}")
    n_space, n_modes = basis.shape
    if n_sensors <= 0 or n_sensors > n_space:
        raise ValueError(
            f"n_sensors={n_sensors} must be in [1, {n_space}]"
        )
    if method != "qr":
        raise NotImplementedError(f"method '{method}' not supported; use 'qr'")

    piv = _greedy_pivot(basis, n_sensors)

    sensors = np.asarray(piv[:n_sensors], dtype=np.intp)
    sensors.sort()
    logger.info("센서 배치 완료: %d/%d 위치 선택 (method=%s)", n_sensors, n_space, method)
    return sensors


def _greedy_pivot(
    basis: NDArray[np.float64], n_sensors: int
) -> NDArray[np.intp]:
    """scipy 없을 때 사용하는 그리디 피벗 근사.

    각 단계에서 아직 선택되지 않은 열 중 현재 잔차와 가장 상관이 높은 위치를 선택.
    """
    n_space, n_modes = basis.shape
    selected: list[int] = []
    residual = basis.copy()

    step = 0
    limit = min(n_sensors, n_modes)
    while step < limit:
        norms = np.linalg.norm(residual, axis=1)
        norms[selected] = -1.0
        idx = int(np.argmax(norms))
        selected.append(idx)
        # 선택된 위치의 기여 제거
        col = residual[idx]
        col_norm = np.dot(col, col)
        if col_norm > 1e-14:
            proj = residual @ col / col_norm
            residual -= np.outer(proj, col)
        step += 1

    # 나머지는 미선택 중 임의로 채움
    remaining = []
    i = 0
    while i < n_space:
        if i not in selected:
            remaining.append(i)
        i += 1
    selected.extend(remaining[: n_sensors - len(selected)])
    return np.array(selected, dtype=np.intp)


def reconstruct(
    basis: NDArray[np.float64],
    sensors: NDArray[np.intp],
    measurements: NDArray[np.float64],
) -> NDArray[np.float64]:
    """센서 측정값으로 전체 공간 상태를 재구성한다.

    선택된 센서 위치에서의 기저 행렬로 최소자승 계수를 구하고,
    전체 기저에 투영해 공간 전체 상태를 복원한다.

    Args:
        basis: (n_space, n_modes) — 공간 기저 행렬.
        sensors: (n_sensors,) — 선택된 센서 위치 인덱스.
        measurements: (n_sensors,) 또는 (n_sensors, n_samples) — 측정값.

    Returns:
        (n_space,) 또는 (n_space, n_samples) — 재구성된 전체 상태.

    Raises:
        ValueError: 입력 형상이 맞지 않는 경우.
    """
    basis = np.asarray(basis, dtype=np.float64)
    sensors = np.asarray(sensors, dtype=np.intp)
    measurements = np.asarray(measurements, dtype=np.float64)

    if basis.ndim != 2:
        raise ValueError(f"basis must be 2D, got {basis.shape}")
    if sensors.ndim != 1:
        raise ValueError(f"sensors must be 1D, got {sensors.shape}")

    squeeze = measurements.ndim == 1
    if squeeze:
        measurements = measurements[:, None]

    if measurements.shape[0] != len(sensors):
        raise ValueError(
            f"measurements.shape[0]={measurements.shape[0]} != len(sensors)={len(sensors)}"
        )

    Theta = basis[sensors, :]  # (n_sensors, n_modes)
    # 최소자승: coeffs = (Theta^T Theta)^{-1} Theta^T y
    coeffs, _, _, _ = np.linalg.lstsq(Theta, measurements, rcond=None)
    reconstruction = basis @ coeffs  # (n_space, n_samples)

    if squeeze:
        return reconstruction[:, 0]
    return reconstruction


__all__ = ["select_sensors", "reconstruct"]
