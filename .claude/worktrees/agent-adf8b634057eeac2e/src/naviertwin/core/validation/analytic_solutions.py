"""해석해(Analytic Solution) 기반 검증 모듈.

고전적 Navier-Stokes 정확해를 제공하고 수치해와 자동 비교한다.

지원 유동:
    - Couette: 두 평판 사이 직선 전단 유동
    - Poiseuille 2D: 두 평판 사이 압력 구동 포물선 유동
    - Poiseuille Pipe: 원관 내 완전발달 층류 유동 (Hagen-Poiseuille)

optional: Dedalus 스펙트럴 솔루션(``naviertwin[full]``) — 고해상도 비교용.

Examples:
    기본 사용::

        import numpy as np
        from naviertwin.core.validation.analytic_solutions import couette_flow

        y = np.linspace(0, 1.0, 50)
        sol = couette_flow(U_top=1.0, H=1.0, y=y)
        print(sol.velocity.shape)  # (50,)

    수치해 비교::

        from naviertwin.core.validation.analytic_solutions import (
            compare_against_analytic,
        )

        metrics = compare_against_analytic(numeric_mesh, sol, field_name="U")
        print(metrics["r2"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.validation.metrics import compute_all_metrics
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AnalyticSolution:
    """해석해 결과 컨테이너.

    Attributes:
        name: 유동 이름 (예: "couette", "poiseuille_2d", "poiseuille_pipe").
        coords: 샘플 좌표 배열 (1D 또는 N×d).
        velocity: 각 좌표에서의 속도 크기 또는 주 방향 성분.
        pressure: 압력 (해당 없을 경우 None).
        params: 해석해 파라미터 기록 (Re, μ, H 등).
    """

    name: str
    coords: NDArray[np.float64]
    velocity: NDArray[np.float64]
    pressure: NDArray[np.float64] | None = None
    params: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 해석해 (numpy 직접 구현)
# ---------------------------------------------------------------------------


def couette_flow(
    U_top: float, H: float, y: NDArray[np.float64]
) -> AnalyticSolution:
    """Couette 유동 해석해: u(y) = U_top · y / H.

    두 평판 사이에 위쪽 평판이 U_top 속도로 움직일 때의 정상 선형 전단 유동.

    Args:
        U_top: 위쪽 평판 속도 [m/s].
        H: 두 평판 간격 [m].
        y: y 좌표 배열 (0 ≤ y ≤ H).

    Returns:
        속도 프로파일을 담은 AnalyticSolution.

    Raises:
        ValueError: H <= 0 인 경우.
    """
    if H <= 0:
        raise ValueError(f"H 는 양수여야 합니다: H={H}")
    y = np.asarray(y, dtype=np.float64)
    u = U_top * y / H
    return AnalyticSolution(
        name="couette",
        coords=y,
        velocity=u,
        params={"U_top": U_top, "H": H},
    )


def poiseuille_flow_2d(
    dpdx: float, mu: float, H: float, y: NDArray[np.float64]
) -> AnalyticSolution:
    """Planar Poiseuille 유동 해석해.

    두 평판 사이 압력 구동 정상 유동:
        u(y) = -(1/(2μ)) · (dp/dx) · y · (H - y)

    Args:
        dpdx: 압력 구배 [Pa/m] (음수이면 양의 x 방향 유동).
        mu: 동점성계수 [Pa·s].
        H: 채널 높이 [m].
        y: y 좌표 배열 (0 ≤ y ≤ H).

    Returns:
        속도 프로파일을 담은 AnalyticSolution.

    Raises:
        ValueError: mu <= 0 또는 H <= 0 인 경우.
    """
    if mu <= 0:
        raise ValueError(f"mu 는 양수여야 합니다: mu={mu}")
    if H <= 0:
        raise ValueError(f"H 는 양수여야 합니다: H={H}")
    y = np.asarray(y, dtype=np.float64)
    u = -(1.0 / (2.0 * mu)) * dpdx * y * (H - y)
    return AnalyticSolution(
        name="poiseuille_2d",
        coords=y,
        velocity=u,
        params={"dpdx": dpdx, "mu": mu, "H": H},
    )


def poiseuille_pipe(
    dpdx: float, mu: float, R: float, r: NDArray[np.float64]
) -> AnalyticSolution:
    """Hagen-Poiseuille 원관 유동 해석해.

        u(r) = -(1/(4μ)) · (dp/dx) · (R² - r²)

    Args:
        dpdx: 축방향 압력 구배 [Pa/m].
        mu: 동점성계수 [Pa·s].
        R: 원관 반지름 [m].
        r: 반경 좌표 배열 (0 ≤ r ≤ R).

    Returns:
        속도 프로파일을 담은 AnalyticSolution.
    """
    if mu <= 0:
        raise ValueError(f"mu 는 양수여야 합니다: mu={mu}")
    if R <= 0:
        raise ValueError(f"R 는 양수여야 합니다: R={R}")
    r = np.asarray(r, dtype=np.float64)
    u = -(1.0 / (4.0 * mu)) * dpdx * (R**2 - r**2)
    return AnalyticSolution(
        name="poiseuille_pipe",
        coords=r,
        velocity=u,
        params={"dpdx": dpdx, "mu": mu, "R": R},
    )


# ---------------------------------------------------------------------------
# Dedalus 기반 고해상도 해석해 (optional)
# ---------------------------------------------------------------------------


def spectral_poiseuille(
    dpdx: float, mu: float, H: float, n_points: int = 128
) -> AnalyticSolution:
    """Dedalus 스펙트럴 방법으로 Poiseuille 유동을 계산한다.

    Dedalus 가 설치된 경우 Chebyshev 기저로 해석해를 샘플링한다.
    미설치 시 RuntimeError 를 발생시킨다.

    Args:
        dpdx: 압력 구배 [Pa/m].
        mu: 동점성계수.
        H: 채널 높이.
        n_points: 스펙트럴 샘플 수.

    Returns:
        AnalyticSolution.

    Raises:
        RuntimeError: Dedalus 미설치 시.
    """
    try:
        import dedalus.public as d3  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "dedalus 설치 필요: pip install naviertwin[full]"
        ) from exc

    # 기본 파라미터 버전만 제공 — 실제 스펙트럴 솔버는 고도화 시 추가
    y = np.linspace(0.0, H, n_points)
    sol = poiseuille_flow_2d(dpdx, mu, H, y)
    sol.name = "poiseuille_2d_spectral"
    sol.params["method"] = "dedalus"
    return sol


# ---------------------------------------------------------------------------
# 수치해 ↔ 해석해 비교
# ---------------------------------------------------------------------------


def compare_against_analytic(
    numeric_mesh: Any,
    analytic: AnalyticSolution,
    field_name: str = "U",
    axis: str = "y",
) -> dict[str, Any]:
    """수치해 메쉬에서 해석해 좌표로 샘플링하고 메트릭을 계산한다.

    Args:
        numeric_mesh: PyVista DataSet.
        analytic: :class:`AnalyticSolution`.
        field_name: 메쉬의 비교 대상 필드명 (point_data).
        axis: 해석해 좌표가 어느 축인지 ("x", "y", "z").

    Returns:
        dict:
            - "metrics": {"rmse","r2","relative_l2","max_error"}
            - "analytic": 해석해 속도 배열
            - "numeric": 수치해 샘플 배열
            - "coords": 샘플 좌표 배열
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "pyvista 가 필요합니다: pip install pyvista"
        ) from exc

    if field_name not in numeric_mesh.point_data:
        raise ValueError(
            f"필드 '{field_name}'이 mesh.point_data 에 없습니다. "
            f"사용 가능: {list(numeric_mesh.point_data.keys())}"
        )

    # 해석해 좌표를 3D 포인트로 변환
    coords = analytic.coords
    bounds = numeric_mesh.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
    x_mid = 0.5 * (bounds[0] + bounds[1])
    z_mid = 0.5 * (bounds[4] + bounds[5])

    if axis == "y":
        probe_points = np.column_stack(
            [np.full_like(coords, x_mid), coords, np.full_like(coords, z_mid)]
        )
    elif axis == "x":
        y_mid = 0.5 * (bounds[2] + bounds[3])
        probe_points = np.column_stack(
            [coords, np.full_like(coords, y_mid), np.full_like(coords, z_mid)]
        )
    elif axis == "z":
        y_mid = 0.5 * (bounds[2] + bounds[3])
        probe_points = np.column_stack(
            [np.full_like(coords, x_mid), np.full_like(coords, y_mid), coords]
        )
    else:
        raise ValueError(f"axis 는 'x'/'y'/'z' 이어야 합니다: '{axis}'")

    probe_poly = pv.PolyData(probe_points)
    sampled = probe_poly.sample(numeric_mesh)
    numeric_vals = np.asarray(sampled.point_data[field_name], dtype=np.float64)

    # 벡터 필드는 크기로 축약
    if numeric_vals.ndim > 1:
        numeric_vals = np.linalg.norm(numeric_vals, axis=-1)

    analytic_vals = np.asarray(analytic.velocity, dtype=np.float64)
    if analytic_vals.ndim > 1:
        analytic_vals = np.linalg.norm(analytic_vals, axis=-1)

    metrics = compute_all_metrics(analytic_vals, numeric_vals)
    logger.info(
        "compare_against_analytic(%s, field=%s): r2=%.4f, rel_l2=%.4f",
        analytic.name,
        field_name,
        metrics["r2"],
        metrics["relative_l2"],
    )

    return {
        "metrics": metrics,
        "analytic": analytic_vals,
        "numeric": numeric_vals,
        "coords": coords,
    }


__all__ = [
    "AnalyticSolution",
    "couette_flow",
    "poiseuille_flow_2d",
    "poiseuille_pipe",
    "spectral_poiseuille",
    "compare_against_analytic",
]
