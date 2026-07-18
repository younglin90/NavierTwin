"""필드 슬라이스 + 라인 프로브 추출.

볼륨 데이터를 임의 평면/라인으로 절단해 후속 분석 가능한 형태로 추출.
정규/비정규 격자 모두 지원 (KDTree 보간).

상용 툴 대응:
    - Tecplot 360: Slice + Extract Line
    - Ansys CFD-Post: Plane / Line / Polyline
    - EnSight: Cut Plane / Particle Trace

Examples:
    >>> import numpy as np
    >>> # 정규 격자 (10×10×10), z=0.5 단면
    >>> field = np.random.default_rng(0).standard_normal((10, 10, 10))
    >>> from naviertwin.core.flow_analysis.slice_extract import slice_axis_aligned
    >>> sl = slice_axis_aligned(field, axis=2, position=4)
    >>> sl.shape
    (10, 10)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def slice_axis_aligned(
    field: NDArray[np.float64],
    axis: int,
    position: int,
) -> NDArray[np.float64]:
    """축 정렬 슬라이스 — 정규 격자에서 한 축의 인덱스 평면 추출.

    Args:
        field: (N1, N2, N3) 또는 더 높은 차원 필드.
        axis: 슬라이스 축 (0, 1, 2 등).
        position: 정수 인덱스.

    Returns:
        해당 축이 제거된 필드 슬라이스.

    Raises:
        ValueError: axis 또는 position 범위 오류.
    """
    field = np.asarray(field, dtype=np.float64)
    if axis < 0 or axis >= field.ndim:
        raise ValueError(f"axis {axis} outside field shape {field.shape}")
    if position < 0 or position >= field.shape[axis]:
        raise ValueError(
            f"position {position} out of range [0, {field.shape[axis]})"
        )
    return np.take(field, position, axis=axis)


def slice_plane_arbitrary(
    points: NDArray[np.float64],
    field: NDArray[np.float64],
    plane_origin: NDArray[np.float64],
    plane_normal: NDArray[np.float64],
    tolerance: float = 1e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """비정규 점 클라우드에서 임의 평면 슬라이스 추출.

    n · (p - p0) = 0 평면에 대해 |거리| < tolerance인 점들 선택.

    Args:
        points: (N, 3) 점 좌표.
        field: (N,) 또는 (N, k) 점당 값.
        plane_origin: (3,) 평면 한 점.
        plane_normal: (3,) 평면 법선 (정규화 안 해도 됨).
        tolerance: 평면 두께.

    Returns:
        (selected_points, selected_field).

    Raises:
        ValueError: 형상 오류.
    """
    points = np.asarray(points, dtype=np.float64)
    field = np.asarray(field, dtype=np.float64)
    plane_origin = np.asarray(plane_origin, dtype=np.float64)
    plane_normal = np.asarray(plane_normal, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points shape must be (N, 3), got {points.shape}")
    if plane_origin.shape != (3,) or plane_normal.shape != (3,):
        raise ValueError(
            f"plane_origin/normal must be (3,), got {plane_origin.shape}, {plane_normal.shape}"
        )
    if field.shape[0] != points.shape[0]:
        raise ValueError(
            f"field length {field.shape[0]} != points {points.shape[0]}"
        )

    n = plane_normal / max(np.linalg.norm(plane_normal), 1e-30)
    dist = (points - plane_origin) @ n
    mask = np.abs(dist) < tolerance
    return points[mask], field[mask]


def line_probe(
    points: NDArray[np.float64],
    field: NDArray[np.float64],
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    n_samples: int = 100,
    method: str = "nearest",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """두 점을 잇는 선분에서 필드 샘플링.

    Args:
        points: (N, 3) 점 좌표.
        field: (N,) 또는 (N, k).
        start, end: (3,) 라인 양 끝점.
        n_samples: 샘플링 점 수.
        method: "nearest" 또는 "idw" (역거리 가중).

    Returns:
        (line_pts, sampled_field): line_pts 형상 (n_samples, 3),
        sampled_field 형상 (n_samples,) 또는 (n_samples, k).

    Raises:
        ValueError: 형상 오류 또는 method 오류.
    """
    points = np.asarray(points, dtype=np.float64)
    field = np.asarray(field, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points shape must be (N, 3), got {points.shape}")
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError(
            f"start/end must be (3,), got {start.shape}, {end.shape}"
        )
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")

    s = np.linspace(0.0, 1.0, n_samples)
    line_pts = start[None, :] + s[:, None] * (end - start)[None, :]

    if method == "nearest":
        sampled = _nearest_neighbor(points, field, line_pts)
    elif method == "idw":
        sampled = _idw(points, field, line_pts, k=4)
    else:
        raise ValueError(f"method '{method}' invalid; use 'nearest'/'idw'")

    return line_pts, sampled


def _nearest_neighbor(
    points: NDArray[np.float64],
    field: NDArray[np.float64],
    queries: NDArray[np.float64],
) -> NDArray[np.float64]:
    """단순 최근접 (작은 데이터용)."""
    distances = np.linalg.norm(
        points[np.newaxis, :, :] - queries[:, np.newaxis, :],
        axis=2,
    )
    return field[np.argmin(distances, axis=1)]


def _idw(
    points: NDArray[np.float64],
    field: NDArray[np.float64],
    queries: NDArray[np.float64],
    k: int = 4,
    p: float = 2.0,
) -> NDArray[np.float64]:
    """Inverse-Distance Weighted (IDW) 보간."""
    n_pts = points.shape[0]
    k = min(k, n_pts)

    distances = np.linalg.norm(
        points[np.newaxis, :, :] - queries[:, np.newaxis, :],
        axis=2,
    )
    idx = np.argpartition(distances, k - 1, axis=1)[:, :k]
    d_k = np.take_along_axis(distances, idx, axis=1)
    zero_mask = d_k < 1e-30
    exact = np.any(zero_mask, axis=1)
    out_shape = (queries.shape[0],) + field.shape[1:]
    out = np.empty(out_shape, dtype=field.dtype)

    if np.any(~exact):
        active = ~exact
        weights = 1.0 / (d_k[active] ** p)
        weights = weights / weights.sum(axis=1, keepdims=True)
        values = field[idx[active]]
        if field.ndim == 1:
            out[active] = np.sum(weights * values, axis=1)
        else:
            out[active] = np.einsum("qk,qk...->q...", weights, values)

    if np.any(exact):
        exact_idx = np.argmax(zero_mask[exact], axis=1)
        out[exact] = field[idx[exact, exact_idx]]
    return out


def polyline_arc_length(line_pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """폴리라인 누적 곡선길이 (시작=0).

    Args:
        line_pts: (N, dim).

    Returns:
        (N,) 누적 호 길이.
    """
    line_pts = np.asarray(line_pts, dtype=np.float64)
    if line_pts.ndim != 2:
        raise ValueError(f"line_pts must be 2D, got {line_pts.shape}")
    diffs = np.linalg.norm(np.diff(line_pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(diffs)])


__all__ = [
    "slice_axis_aligned",
    "slice_plane_arbitrary",
    "line_probe",
    "polyline_arc_length",
]
