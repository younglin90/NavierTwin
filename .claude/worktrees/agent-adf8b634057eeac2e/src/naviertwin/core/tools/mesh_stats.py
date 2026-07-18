"""메쉬 기본 통계 — bbox / 셀 수 / 평균 edge length (PyVista 기반).

Examples:
    >>> import pyvista as pv  # doctest: +SKIP
    >>> from naviertwin.core.tools.mesh_stats import mesh_stats  # doctest: +SKIP
    >>> mesh_stats(pv.Cube())  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def bounding_box(points: NDArray[np.float64]) -> dict[str, float]:
    """points (n, 3) → bbox."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points (n, 3) 필요")
    return {
        "xmin": float(p[:, 0].min()), "xmax": float(p[:, 0].max()),
        "ymin": float(p[:, 1].min()), "ymax": float(p[:, 1].max()),
        "zmin": float(p[:, 2].min()), "zmax": float(p[:, 2].max()),
        "dx": float(p[:, 0].max() - p[:, 0].min()),
        "dy": float(p[:, 1].max() - p[:, 1].min()),
        "dz": float(p[:, 2].max() - p[:, 2].min()),
        "diagonal": float(np.linalg.norm(p.max(axis=0) - p.min(axis=0))),
    }


def edge_length_stats(
    points: NDArray[np.float64], edges: NDArray[np.int64] | None = None,
) -> dict[str, float]:
    """edges (n_edges, 2) 인덱스에 대한 edge length 통계.
    edges=None 이면 point pairs 를 샘플링해 glancing 통계 반환."""
    p = np.asarray(points, dtype=np.float64)
    if edges is None:
        # 랜덤 샘플링
        rng = np.random.default_rng(0)
        n = min(1000, p.shape[0])
        i = rng.integers(0, p.shape[0], n)
        j = rng.integers(0, p.shape[0], n)
        d = np.linalg.norm(p[i] - p[j], axis=1)
    else:
        e = np.asarray(edges)
        d = np.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=1)
    return {
        "mean": float(d.mean()),
        "std": float(d.std()),
        "min": float(d.min()),
        "max": float(d.max()),
        "p05": float(np.quantile(d, 0.05)),
        "p95": float(np.quantile(d, 0.95)),
    }


def mesh_stats(mesh: Any) -> dict[str, Any]:
    """PyVista Dataset 의 요약 통계."""
    try:
        import pyvista as pv  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pyvista 필요") from exc
    pts = np.asarray(mesh.points, dtype=np.float64)
    out: dict[str, Any] = {
        "n_points": int(mesh.n_points),
        "n_cells": int(mesh.n_cells),
        "bbox": bounding_box(pts),
    }
    try:
        out["volume"] = float(mesh.volume)
    except Exception:  # noqa: BLE001
        out["volume"] = None
    return out


__all__ = ["bounding_box", "edge_length_stats", "mesh_stats"]
