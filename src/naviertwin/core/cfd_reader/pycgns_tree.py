"""pyCGNS CGNS.MAP 트리 직접 노출 — hierarchical node 탐색.

기존 CGNS reader 는 pyvista→pyCGNS→h5py→meshio 폴백 체인. 이 모듈은 pyCGNS 의
CGNS 트리 자체를 사용자가 직접 탐색할 수 있게 한다.

Examples:
    >>> from naviertwin.core.cfd_reader.pycgns_tree import (
    ...     load_cgns_tree, list_zones, list_solutions,
    ... )
    >>> # tree = load_cgns_tree("mesh.cgns")
    >>> # zones = list_zones(tree)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _require_pycgns() -> Any:
    try:
        from CGNS import MAP as cgmap

        return cgmap
    except ImportError as exc:
        raise RuntimeError("pyCGNS 필요: pip install pyCGNS") from exc


def load_cgns_tree(path: str | Path) -> Any:
    """CGNS 파일을 메모리 트리로 로드."""
    cgmap = _require_pycgns()
    tree, _, _ = cgmap.load(str(path))
    return tree


def walk_tree(tree: Any, depth: int = 0, max_depth: int = 10) -> list[str]:
    """트리 노드를 넓이우선 탐색해 이름/타입 리스트 반환."""
    names: list[str] = []

    def _recurse(node: Any, level: int) -> None:
        if level > max_depth:
            return
        if not isinstance(node, list) or len(node) < 4:
            return
        name = node[0]
        ntype = node[3]
        names.append(f"{'  ' * level}{name} [{ntype}]")
        children = node[2]
        if isinstance(children, list):
            child_idx = 0
            while child_idx < len(children):
                _recurse(children[child_idx], level + 1)
                child_idx += 1

    _recurse(tree, depth)
    return names


def list_zones(tree: Any) -> list[str]:
    """CGNSBase_t 아래 Zone_t 노드 이름 리스트."""
    zones: list[str] = []
    if not isinstance(tree, list) or len(tree) < 3:
        return zones

    bases = tree[2] if isinstance(tree[2], list) else []
    base_idx = 0
    while base_idx < len(bases):
        base = bases[base_idx]
        if not (isinstance(base, list) and len(base) >= 4):
            base_idx += 1
            continue
        if base[3] != "CGNSBase_t":
            base_idx += 1
            continue
        zone_nodes = base[2] if isinstance(base[2], list) else []
        zone_idx = 0
        while zone_idx < len(zone_nodes):
            z = zone_nodes[zone_idx]
            if isinstance(z, list) and len(z) >= 4 and z[3] == "Zone_t":
                zones.append(z[0])
            zone_idx += 1
        base_idx += 1
    return zones


def list_solutions(tree: Any) -> list[tuple[str, str]]:
    """각 Zone 내부의 FlowSolution_t 노드 (zone, solution) 리스트."""
    out: list[tuple[str, str]] = []
    if not isinstance(tree, list) or len(tree) < 3:
        return out

    bases = tree[2] if isinstance(tree[2], list) else []
    base_idx = 0
    while base_idx < len(bases):
        base = bases[base_idx]
        if not (isinstance(base, list) and base[3] == "CGNSBase_t"):
            base_idx += 1
            continue
        zone_nodes = base[2] if isinstance(base[2], list) else []
        zone_idx = 0
        while zone_idx < len(zone_nodes):
            z = zone_nodes[zone_idx]
            if not (isinstance(z, list) and z[3] == "Zone_t"):
                zone_idx += 1
                continue
            zone_name = z[0]
            solution_nodes = z[2] if isinstance(z[2], list) else []
            sol_idx = 0
            while sol_idx < len(solution_nodes):
                s = solution_nodes[sol_idx]
                if isinstance(s, list) and s[3] == "FlowSolution_t":
                    out.append((zone_name, s[0]))
                sol_idx += 1
            zone_idx += 1
        base_idx += 1
    return out


def get_coordinates(tree: Any, zone_name: str) -> dict[str, Any]:
    """특정 Zone 의 CoordinateX/Y/Z 배열을 딕셔너리로 반환."""
    import numpy as np

    coords: dict[str, Any] = {}
    bases = tree[2] if isinstance(tree[2], list) else []
    base_idx = 0
    while base_idx < len(bases):
        base = bases[base_idx]
        if base[3] != "CGNSBase_t":
            base_idx += 1
            continue
        zone_nodes = base[2] if isinstance(base[2], list) else []
        zone_idx = 0
        while zone_idx < len(zone_nodes):
            z = zone_nodes[zone_idx]
            if z[3] != "Zone_t" or z[0] != zone_name:
                zone_idx += 1
                continue
            grid_nodes = z[2] if isinstance(z[2], list) else []
            grid_idx = 0
            while grid_idx < len(grid_nodes):
                gc = grid_nodes[grid_idx]
                if gc[3] != "GridCoordinates_t":
                    grid_idx += 1
                    continue
                coord_nodes = gc[2] if isinstance(gc[2], list) else []
                coord_idx = 0
                while coord_idx < len(coord_nodes):
                    c = coord_nodes[coord_idx]
                    if c[3] == "DataArray_t":
                        coords[c[0]] = np.asarray(c[1])
                    coord_idx += 1
                grid_idx += 1
            zone_idx += 1
        base_idx += 1
    return coords


__all__ = [
    "load_cgns_tree",
    "walk_tree",
    "list_zones",
    "list_solutions",
    "get_coordinates",
]
