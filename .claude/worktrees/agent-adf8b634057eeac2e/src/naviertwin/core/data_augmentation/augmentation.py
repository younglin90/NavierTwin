"""물리 기반 데이터 증강 모듈.

갈릴레이 불변성, 공간 대칭(반사/회전), 스케일링 증강으로 CFD 스냅샷을
확장한다. Surrogate / Operator Learning 학습에 활용.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_augmentation.augmentation import (
    ...     galilean_shift, reflect, rotate_2d, scale_field,
    ... )
    >>> U = np.random.default_rng(0).standard_normal((100, 3))
    >>> U2 = galilean_shift(U, np.array([1.0, 0.0, 0.0]))
    >>> U3 = reflect(U, axis=1)
    >>> U4 = rotate_2d(U, angle_rad=np.pi / 4)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def galilean_shift(
    U: NDArray[np.float64], u_ref: NDArray[np.float64]
) -> NDArray[np.float64]:
    """속도장에 일정 속도 u_ref 를 더한다.

    Args:
        U: 속도 필드. shape = (..., d) (d=2 또는 3).
        u_ref: 기준 속도. shape = (d,).

    Returns:
        U + u_ref.
    """
    U = np.asarray(U, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64)
    return U + u_ref


def reflect(
    U: NDArray[np.float64], axis: int
) -> NDArray[np.float64]:
    """주어진 공간 축에 대해 속도 벡터를 반사.

    반사: 해당 성분의 부호를 뒤집는다 (자유 공간 대칭).

    Args:
        U: shape = (..., d).
        axis: 반사할 축 (0=x, 1=y, 2=z).

    Returns:
        반사된 속도 필드.
    """
    U = np.asarray(U, dtype=np.float64).copy()
    U[..., axis] = -U[..., axis]
    return U


def rotate_2d(
    U: NDArray[np.float64], angle_rad: float
) -> NDArray[np.float64]:
    """xy 평면에서 2D 속도를 회전.

    Args:
        U: shape = (..., 2) 또는 (..., 3) (z 성분은 보존).
        angle_rad: 회전각 [rad].

    Returns:
        회전된 속도 필드.
    """
    U = np.asarray(U, dtype=np.float64).copy()
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    ux = U[..., 0].copy()
    uy = U[..., 1].copy()
    U[..., 0] = c * ux - s * uy
    U[..., 1] = s * ux + c * uy
    return U


def scale_field(
    field: NDArray[np.float64], factor: float
) -> NDArray[np.float64]:
    """필드 스케일링 (균일 배수).

    Args:
        field: 임의 shape.
        factor: 곱할 배수.
    """
    return np.asarray(field, dtype=np.float64) * float(factor)


def augment_symmetric(
    U: NDArray[np.float64], axes: tuple[int, ...] = (0, 1)
) -> list[NDArray[np.float64]]:
    """모든 대칭 반사 조합을 반환한다 (자기 포함).

    Args:
        U: shape = (..., d).
        axes: 반사할 축 집합.

    Returns:
        2^len(axes) 개의 증강 필드 리스트.
    """
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by augment_symmetric")
    return _kernels.augment_symmetric(np.asarray(U, dtype=np.float64), axes)


__all__ = [
    "galilean_shift",
    "reflect",
    "rotate_2d",
    "scale_field",
    "augment_symmetric",
]
