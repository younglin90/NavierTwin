"""벡터장 위상학적 임계점 검출 + 분류.

2D 벡터장에서 |U| → 0이 되는 점을 찾고, Jacobian의 고유값으로
saddle/sink/source/spiral 분류. 상용 후처리 툴의 Vector Field Topology
기능. 와류 핵심 위치 자동 식별.

References:
    Helman & Hesselink, "Visualizing Vector Field Topology in Fluid Flows",
    IEEE Comp. Graphics & Applications, 1991.

Examples:
    >>> import numpy as np
    >>> # 회전장: u = -y, v = x → center at (0, 0)
    >>> x = np.linspace(-1, 1, 21)
    >>> X, Y = np.meshgrid(x, x, indexing="ij")
    >>> u, v = -Y, X
    >>> from naviertwin.core.flow_analysis.critical_points import find_critical_points
    >>> cps = find_critical_points(u, v, dx=0.1, dy=0.1)
    >>> len(cps) >= 1
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def classify_2d(
    jacobian: NDArray[np.float64],
) -> str:
    """Jacobian 고유값으로 임계점 분류.

    Categories:
        - saddle:           Re(λ₁)·Re(λ₂) < 0
        - source / repelling node: 두 실수 양수 고유값
        - sink / attracting node:  두 실수 음수 고유값
        - spiral_source / repelling focus: 복소 고유값, Re > 0
        - spiral_sink / attracting focus:  복소 고유값, Re < 0
        - center:           순수 허수 고유값
        - degenerate:       0에 가까운 고유값

    Args:
        jacobian: (2, 2) 야코비 행렬.

    Returns:
        분류 문자열.

    Raises:
        ValueError: 형상 오류.
    """
    J = np.asarray(jacobian, dtype=np.float64)
    if J.shape != (2, 2):
        raise ValueError(f"jacobian must be (2, 2), got {J.shape}")

    eigvals = np.linalg.eigvals(J)
    re = np.real(eigvals)
    im = np.imag(eigvals)

    # degenerate
    if np.any(np.abs(eigvals) < 1e-12):
        return "degenerate"

    # complex pair
    if np.any(np.abs(im) > 1e-9):
        if np.all(np.abs(re) < 1e-9):
            return "center"
        if np.all(re > 0):
            return "spiral_source"
        if np.all(re < 0):
            return "spiral_sink"
        return "spiral_saddle"

    # real eigenvalues
    if re[0] * re[1] < 0:
        return "saddle"
    if np.all(re > 0):
        return "source"
    if np.all(re < 0):
        return "sink"
    return "degenerate"


def find_critical_points(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
    tol: float = 1e-6,
) -> list[dict]:
    """2D 벡터장의 임계점 검출 (영점 위치 + 분류).

    셀별로 4개 모서리 부호가 바뀌면 영점이 셀 내부에 존재.
    셀 내부 위치는 이중선형 보간으로 정밀화.

    Args:
        u: (Nx, Ny) x 속도 성분.
        v: (Nx, Ny) y 속도 성분.
        dx, dy: 격자 간격.
        tol: 영점 판정 허용 오차.

    Returns:
        list of {"x": float, "y": float, "type": str, "i": int, "j": int}.

    Raises:
        ValueError: u/v 형상 불일치.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError(
            f"u/v must be same-shape 2D, got {u.shape} vs {v.shape}"
        )

    nx, ny = u.shape
    crits: list[dict] = []

    i = 0
    while i < nx - 1:
        j = 0
        while j < ny - 1:
            # 셀 4개 모서리
            uc = np.array([u[i, j], u[i + 1, j], u[i + 1, j + 1], u[i, j + 1]])
            vc = np.array([v[i, j], v[i + 1, j], v[i + 1, j + 1], v[i, j + 1]])
            # 셀 내 영점 가능 조건: u/v 모두 min*max ≤ 0
            if uc.min() * uc.max() > tol * tol:
                j += 1
                continue
            if vc.min() * vc.max() > tol * tol:
                j += 1
                continue

            # 셀 내 단순 이중선형 보간 영점 (중심 근사)
            local_x = 0.5
            local_y = 0.5
            refine_iter = 0
            while refine_iter < 20:
                # bilinear interp
                u_eval = ((1 - local_x) * (1 - local_y) * uc[0]
                          + local_x * (1 - local_y) * uc[1]
                          + local_x * local_y * uc[2]
                          + (1 - local_x) * local_y * uc[3])
                v_eval = ((1 - local_x) * (1 - local_y) * vc[0]
                          + local_x * (1 - local_y) * vc[1]
                          + local_x * local_y * vc[2]
                          + (1 - local_x) * local_y * vc[3])
                if abs(u_eval) < tol and abs(v_eval) < tol:
                    break
                # 야코비
                du_dx = (1 - local_y) * (uc[1] - uc[0]) + local_y * (uc[2] - uc[3])
                du_dy = (1 - local_x) * (uc[3] - uc[0]) + local_x * (uc[2] - uc[1])
                dv_dx = (1 - local_y) * (vc[1] - vc[0]) + local_y * (vc[2] - vc[3])
                dv_dy = (1 - local_x) * (vc[3] - vc[0]) + local_x * (vc[2] - vc[1])
                J = np.array([[du_dx, du_dy], [dv_dx, dv_dy]])
                det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                if abs(det) < 1e-30:
                    break
                # Newton step
                inv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / det
                step = inv @ np.array([u_eval, v_eval])
                local_x -= step[0]
                local_y -= step[1]
                # bounds
                if not (0.0 <= local_x <= 1.0 and 0.0 <= local_y <= 1.0):
                    break
                refine_iter += 1

            if not (0.0 <= local_x <= 1.0 and 0.0 <= local_y <= 1.0):
                j += 1
                continue
            if abs(u_eval) > 10 * tol or abs(v_eval) > 10 * tol:
                j += 1
                continue

            # 분류
            du_dx = (1 - local_y) * (uc[1] - uc[0]) + local_y * (uc[2] - uc[3])
            du_dy = (1 - local_x) * (uc[3] - uc[0]) + local_x * (uc[2] - uc[1])
            dv_dx = (1 - local_y) * (vc[1] - vc[0]) + local_y * (vc[2] - vc[3])
            dv_dy = (1 - local_x) * (vc[3] - vc[0]) + local_x * (vc[2] - vc[1])
            J = np.array([[du_dx / dx, du_dy / dy], [dv_dx / dx, dv_dy / dy]])
            cp_type = classify_2d(J)

            crits.append({
                "x": (i + local_x) * dx,
                "y": (j + local_y) * dy,
                "type": cp_type,
                "i": i,
                "j": j,
            })
            j += 1
        i += 1

    logger.info("임계점 검출 완료: %d개", len(crits))
    return crits


def poincare_index(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    center_i: int,
    center_j: int,
) -> int:
    """주어진 (i, j) 주변 4점 회로에서 Poincaré 지수 계산.

    원형 적분 ∮ (dθ) / 2π. saddle = -1, vortex = +1.

    Args:
        u, v: 벡터장.
        center_i, center_j: 중심 인덱스.

    Returns:
        Poincaré 지수 (정수).

    Raises:
        ValueError: 인덱스 범위 오류.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nx, ny = u.shape
    if not (1 <= center_i < nx - 1 and 1 <= center_j < ny - 1):
        raise ValueError(
            f"center ({center_i}, {center_j}) too close to boundary "
            f"with shape {u.shape}"
        )

    # 4 corners, CCW
    pts = np.array([
        (center_i + 1, center_j),
        (center_i + 1, center_j + 1),
        (center_i - 1, center_j + 1),
        (center_i - 1, center_j),
        (center_i - 1, center_j - 1),
        (center_i + 1, center_j - 1),
    ], dtype=np.intp)
    angles = np.arctan2(v[pts[:, 0], pts[:, 1]], u[pts[:, 0], pts[:, 1]])
    diffs = np.roll(angles, -1) - angles
    total = float(np.sum((diffs + np.pi) % (2 * np.pi) - np.pi))
    return int(round(total / (2.0 * np.pi)))


__all__ = ["classify_2d", "find_critical_points", "poincare_index"]
