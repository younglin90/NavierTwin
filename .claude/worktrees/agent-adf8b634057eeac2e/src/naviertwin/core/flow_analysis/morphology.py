"""필드 형태론 (Morphology) — ROI 마스크의 이진 연산.

CFD 후처리에서 ROI(Region of Interest) 정의, 와류 코어 영역 식별,
임계값 기반 필드 분할, connected component 라벨링.

상용 툴 대응:
    - SciPy ndimage: binary_dilation, binary_erosion, label
    - MATLAB: imdilate, imerode, bwlabel
    - Tecplot 360: Region of Interest filter
    - 학술: Serra, J., "Image Analysis and Mathematical Morphology", 1982.

Examples:
    >>> import numpy as np
    >>> mask = np.zeros((10, 10), dtype=bool)
    >>> mask[3:7, 3:7] = True
    >>> from naviertwin.core.flow_analysis.morphology import binary_dilation_2d
    >>> dilated = binary_dilation_2d(mask)
    >>> dilated.sum() > mask.sum()
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def binary_dilation_2d(
    mask: NDArray[np.bool_],
    iterations: int = 1,
    connectivity: int = 1,
) -> NDArray[np.bool_]:
    """이진 마스크 팽창 (Dilation).

    Args:
        mask: (Nx, Ny) 부울 마스크.
        iterations: 반복 횟수.
        connectivity: 1 = 4-neighbor, 2 = 8-neighbor.

    Returns:
        팽창된 마스크.

    Raises:
        ValueError: connectivity 또는 iterations 오류.
    """
    if connectivity not in (1, 2):
        raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")

    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by binary_dilation_2d")
    return _kernels.binary_dilation_2d(np.asarray(mask, dtype=bool), iterations, connectivity)


def binary_erosion_2d(
    mask: NDArray[np.bool_],
    iterations: int = 1,
    connectivity: int = 1,
) -> NDArray[np.bool_]:
    """이진 마스크 침식 (Erosion).

    경계 외부는 False(배경)으로 간주 — 가장자리 픽셀은 자동 침식.

    Args:
        mask: 부울 마스크.
        iterations: 반복.
        connectivity: 1 또는 2.

    Returns:
        침식된 마스크.

    Raises:
        ValueError: 매개변수 오류.
    """
    if connectivity not in (1, 2):
        raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")

    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by binary_erosion_2d")
    return _kernels.binary_erosion_2d(np.asarray(mask, dtype=bool), iterations, connectivity)


def binary_opening_2d(
    mask: NDArray[np.bool_],
    iterations: int = 1,
    connectivity: int = 1,
) -> NDArray[np.bool_]:
    """Opening: erosion 후 dilation. 작은 노이즈 영역 제거.

    Args:
        mask, iterations, connectivity: 동일.

    Returns:
        Opening된 마스크.
    """
    return binary_dilation_2d(
        binary_erosion_2d(mask, iterations, connectivity),
        iterations, connectivity,
    )


def binary_closing_2d(
    mask: NDArray[np.bool_],
    iterations: int = 1,
    connectivity: int = 1,
) -> NDArray[np.bool_]:
    """Closing: dilation 후 erosion. 작은 구멍 메움."""
    return binary_erosion_2d(
        binary_dilation_2d(mask, iterations, connectivity),
        iterations, connectivity,
    )


def connected_components_2d(
    mask: NDArray[np.bool_],
    connectivity: int = 1,
) -> tuple[NDArray[np.int32], int]:
    """연결 성분 라벨링 — flood-fill 기반.

    Args:
        mask: 부울 마스크.
        connectivity: 1 (4-neighbor) 또는 2 (8-neighbor).

    Returns:
        (labels, n_components): labels 형상 같음, 0 = 배경.

    Raises:
        ValueError: connectivity 오류.
    """
    if connectivity not in (1, 2):
        raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")

    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by connected_components_2d")
    return _kernels.connected_components_2d(np.asarray(mask, dtype=bool), connectivity)


def component_sizes(
    labels: NDArray[np.int32],
) -> NDArray[np.int64]:
    """각 connected component의 픽셀 수 반환.

    Args:
        labels: connected_components_2d 출력.

    Returns:
        (n_components,) 픽셀 수. 인덱스 0 = label 1 크기.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return np.array([], dtype=np.int64)
    n = int(labels.max())
    if n == 0:
        return np.array([], dtype=np.int64)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    return sizes[1:]  # 0번은 배경 제외


def remove_small_components(
    mask: NDArray[np.bool_],
    min_size: int,
    connectivity: int = 1,
) -> NDArray[np.bool_]:
    """크기 < min_size 인 연결 성분 제거.

    Args:
        mask: 부울 마스크.
        min_size: 최소 픽셀 수.
        connectivity: 1 또는 2.

    Returns:
        필터링된 마스크.

    Raises:
        ValueError: min_size < 1.
    """
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}")
    labels, n = connected_components_2d(mask, connectivity)
    sizes = component_sizes(labels)
    keep = (sizes >= min_size).astype(int)
    keep_arr = np.concatenate([[0], keep])  # label 0 (배경) → 0
    return keep_arr[labels].astype(bool)


def threshold_to_mask(
    field: NDArray[np.float64],
    threshold: float,
    mode: str = "above",
) -> NDArray[np.bool_]:
    """필드를 임계값 기반 이진 마스크로 변환.

    Args:
        field: 임의 형상.
        threshold: 임계값.
        mode: "above", "below", "equal", "abs_above".

    Returns:
        같은 형상의 부울 마스크.

    Raises:
        ValueError: mode 오류.
    """
    field = np.asarray(field, dtype=np.float64)
    if mode == "above":
        return field > threshold
    if mode == "below":
        return field < threshold
    if mode == "equal":
        return field == threshold
    if mode == "abs_above":
        return np.abs(field) > threshold
    raise ValueError(f"mode '{mode}' invalid")


__all__ = [
    "binary_dilation_2d",
    "binary_erosion_2d",
    "binary_opening_2d",
    "binary_closing_2d",
    "connected_components_2d",
    "component_sizes",
    "remove_small_components",
    "threshold_to_mask",
]
