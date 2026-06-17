"""Gmsh Python API 기반 파라미터 메쉬 생성 모듈.

채널(직사각), 원통(2D/3D), 익형(NACA 4-digit) 파라미터 메쉬를 생성한다.
모두 PyVista UnstructuredGrid 로 반환된다.

의존성:
    - gmsh (optional, ``pip install naviertwin[full]``)
    - meshio (core)
    - pyvista (core)

Examples:
    채널 메쉬 생성::

        from naviertwin.core.tools.mesh_generator import generate_channel
        mesh = generate_channel(length=4.0, height=1.0, nx=40, ny=10)
        mesh.plot()
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

_GMSH_MISSING = (
    "gmsh 설치 필요: pip install naviertwin[full]\n"
    "또는 직접 설치: pip install gmsh"
)


def _require_gmsh() -> Any:
    """gmsh 모듈을 import 하거나 친절한 에러를 발생시킨다."""
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError(_GMSH_MISSING) from exc
    return gmsh


def _msh_to_pv_ug(msh_path: Path) -> Any:
    """.vtk/.msh 파일을 PyVista UnstructuredGrid 로 변환한다.

    gmsh 가 .vtk 로 내보낸 경우 pyvista.read 로 직접 읽고,
    .msh 인 경우 meshio 폴백을 사용한다.
    """
    import pyvista as pv

    if msh_path.suffix.lower() in {".vtk", ".vtu"}:
        ug = pv.read(str(msh_path))
    else:
        import meshio

        mesh = meshio.read(str(msh_path))
        ug = pv.from_meshio(mesh)
    if not isinstance(ug, pv.UnstructuredGrid):
        ug = ug.cast_to_unstructured_grid()
    return ug


def generate_channel(
    length: float = 4.0,
    height: float = 1.0,
    nx: int = 40,
    ny: int = 10,
) -> Any:
    """직사각 2D 채널 메쉬(구조격자)를 생성한다.

    Args:
        length: x 방향 길이.
        height: y 방향 높이.
        nx: x 방향 셀 수.
        ny: y 방향 셀 수.

    Returns:
        PyVista UnstructuredGrid (quad 셀).
    """
    gmsh = _require_gmsh()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "channel.vtk"
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("channel")

            lc = min(length / nx, height / ny)
            p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
            p2 = gmsh.model.geo.addPoint(length, 0.0, 0.0, lc)
            p3 = gmsh.model.geo.addPoint(length, height, 0.0, lc)
            p4 = gmsh.model.geo.addPoint(0.0, height, 0.0, lc)

            l1 = gmsh.model.geo.addLine(p1, p2)
            l2 = gmsh.model.geo.addLine(p2, p3)
            l3 = gmsh.model.geo.addLine(p3, p4)
            l4 = gmsh.model.geo.addLine(p4, p1)

            # 구조격자화를 위한 transfinite
            gmsh.model.geo.mesh.setTransfiniteCurve(l1, nx + 1)
            gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx + 1)
            gmsh.model.geo.mesh.setTransfiniteCurve(l2, ny + 1)
            gmsh.model.geo.mesh.setTransfiniteCurve(l4, ny + 1)

            cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
            s1 = gmsh.model.geo.addPlaneSurface([cl])
            gmsh.model.geo.mesh.setTransfiniteSurface(s1)
            gmsh.model.geo.mesh.setRecombine(2, s1)
            gmsh.model.geo.synchronize()

            gmsh.model.mesh.generate(2)
            gmsh.write(str(out))
        finally:
            gmsh.finalize()

        logger.info("채널 메쉬 생성 완료: %.1fx%.1f (%dx%d)", length, height, nx, ny)
        return _msh_to_pv_ug(out)


def generate_cylinder(
    radius: float = 0.5,
    length: float = 2.0,
    n_circum: int = 32,
    n_axial: int = 20,
) -> Any:
    """3D 원통 메쉬(축대칭)를 생성한다.

    Args:
        radius: 원통 반지름.
        length: 축 방향 길이.
        n_circum: 원주 방향 분할 수.
        n_axial: 축 방향 분할 수.

    Returns:
        PyVista UnstructuredGrid (tetrahedral).
    """
    gmsh = _require_gmsh()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "cylinder.vtk"
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("cylinder")

            lc = 2 * math.pi * radius / max(n_circum, 4)
            gmsh.model.occ.addCylinder(
                0.0, 0.0, 0.0, length, 0.0, 0.0, radius
            )
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 0.5)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
            gmsh.option.setNumber(
                "Mesh.CharacteristicLengthExtendFromBoundary", 1
            )

            gmsh.model.mesh.generate(3)
            gmsh.write(str(out))
        finally:
            gmsh.finalize()

        logger.info(
            "원통 메쉬 생성 완료: r=%.2f L=%.2f (circum=%d, axial=%d)",
            radius,
            length,
            n_circum,
            n_axial,
        )
        return _msh_to_pv_ug(out)


def generate_airfoil(
    naca_code: str = "0012",
    chord: float = 1.0,
    n_points: int = 80,
    farfield_radius: float = 20.0,
) -> Any:
    """NACA 4-digit 익형 주위 2D 메쉬(O-type)를 생성한다.

    Args:
        naca_code: NACA 4-digit 코드 문자열 (예: "0012", "2412").
        chord: 익형 코드 길이.
        n_points: 익형 윤곽 점 수.
        farfield_radius: 원거리 경계 반지름 (chord 단위 배수).

    Returns:
        PyVista UnstructuredGrid (triangular).
    """
    gmsh = _require_gmsh()
    if len(naca_code) != 4 or not naca_code.isdigit():
        raise ValueError(f"NACA 4-digit 코드가 필요합니다: '{naca_code}'")

    m = int(naca_code[0]) / 100.0
    p = int(naca_code[1]) / 10.0
    t = int(naca_code[2:]) / 100.0

    def _naca_surface(n: int) -> list[tuple[float, float]]:
        """NACA 4-digit 윤곽 점을 반환한다 (위/아래 통합, counter-clockwise)."""
        x = list(map(lambda i: 0.5 * (1 - math.cos(math.pi * i / n)), range(n + 1)))
        upper: list[tuple[float, float]] = []
        lower: list[tuple[float, float]] = []
        x_idx = 0
        while x_idx < len(x):
            xi = x[x_idx]
            yt = (t / 0.2) * (
                0.2969 * math.sqrt(xi)
                - 0.1260 * xi
                - 0.3516 * xi**2
                + 0.2843 * xi**3
                - 0.1015 * xi**4
            )
            if p > 0 and 0 < xi < 1:
                if xi < p:
                    yc = (m / p**2) * (2 * p * xi - xi**2)
                    dyc = (2 * m / p**2) * (p - xi)
                else:
                    yc = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * xi - xi**2)
                    dyc = (2 * m / (1 - p) ** 2) * (p - xi)
                theta = math.atan(dyc)
                upper.append(
                    (xi - yt * math.sin(theta), yc + yt * math.cos(theta))
                )
                lower.append(
                    (xi + yt * math.sin(theta), yc - yt * math.cos(theta))
                )
            else:
                upper.append((xi, yt))
                lower.append((xi, -yt))
            x_idx += 1
        # counter-clockwise: lower (back → front) + upper (front → back)
        return list(reversed(lower)) + upper[1:]

    pts = list(map(lambda point: (point[0] * chord, point[1] * chord), _naca_surface(n_points // 2)))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "airfoil.vtk"
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("airfoil")

            lc_air = chord / max(n_points, 20)
            lc_far = farfield_radius * chord / 20.0

            airfoil_tags: list[int] = []
            point_idx = 0
            while point_idx < len(pts):
                x, y = pts[point_idx]
                airfoil_tags.append(
                    gmsh.model.geo.addPoint(x, y, 0.0, lc_air)
                )
                point_idx += 1

            airfoil_lines: list[int] = []
            i = 0
            while i < len(airfoil_tags):
                a = airfoil_tags[i]
                b = airfoil_tags[(i + 1) % len(airfoil_tags)]
                airfoil_lines.append(gmsh.model.geo.addLine(a, b))
                i += 1

            R = farfield_radius * chord
            cx, cy = 0.5 * chord, 0.0
            f1 = gmsh.model.geo.addPoint(cx + R, cy, 0.0, lc_far)
            f2 = gmsh.model.geo.addPoint(cx, cy + R, 0.0, lc_far)
            f3 = gmsh.model.geo.addPoint(cx - R, cy, 0.0, lc_far)
            f4 = gmsh.model.geo.addPoint(cx, cy - R, 0.0, lc_far)
            cc = gmsh.model.geo.addPoint(cx, cy, 0.0, lc_far)

            arc1 = gmsh.model.geo.addCircleArc(f1, cc, f2)
            arc2 = gmsh.model.geo.addCircleArc(f2, cc, f3)
            arc3 = gmsh.model.geo.addCircleArc(f3, cc, f4)
            arc4 = gmsh.model.geo.addCircleArc(f4, cc, f1)

            outer = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])
            inner = gmsh.model.geo.addCurveLoop(airfoil_lines)
            gmsh.model.geo.addPlaneSurface([outer, inner])
            gmsh.model.geo.synchronize()

            gmsh.model.mesh.generate(2)
            gmsh.write(str(out))
        finally:
            gmsh.finalize()

        logger.info(
            "NACA%s 익형 메쉬 생성 완료 (chord=%.2f, R=%.1f)",
            naca_code,
            chord,
            farfield_radius,
        )
        return _msh_to_pv_ug(out)


__all__ = ["generate_channel", "generate_cylinder", "generate_airfoil"]
