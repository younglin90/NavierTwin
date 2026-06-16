"""Probe 시계열 — 주어진 점 위치에서 시간축 스냅샷 샘플링.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.probe import probe_time_series
    >>> # snapshots (n_points, n_times), coords (n_points, 3), probes (k, 3)
    >>> X = np.arange(12).reshape(3, 4).astype(float)
    >>> coords = np.array([[0,0,0],[1,0,0],[2,0,0]], dtype=float)
    >>> probes = np.array([[1.0, 0, 0]])
    >>> ts = probe_time_series(X, coords, probes)
    >>> ts.shape
    (1, 4)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by probe analysis")


def probe_time_series(
    snapshots: NDArray[np.float64],
    coords: NDArray[np.float64],
    probes: NDArray[np.float64],
    *,
    method: str = "nearest",
    k: int = 4,
) -> NDArray[np.float64]:
    """각 probe 에 대해 시간축 시계열을 반환.

    Args:
        snapshots: (n_points, n_times).
        coords: (n_points, 3).
        probes: (n_probes, 3).
        method: "nearest" / "idw".
        k: IDW 이웃 개수.

    Returns:
        (n_probes, n_times).
    """
    snapshots = np.asarray(snapshots, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    probes = np.asarray(probes, dtype=np.float64)
    if method not in {"nearest", "idw"}:
        raise ValueError(f"unknown method: {method}")
    return _kernels.probe_time_series(snapshots, coords, probes, method, int(k))


def probe_statistics(time_series: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    """probe 시계열 → mean / std / min / max / rms (probe 축 기준)."""
    x = np.asarray(time_series, dtype=np.float64)
    return {
        "mean": x.mean(axis=1),
        "std": x.std(axis=1),
        "min": x.min(axis=1),
        "max": x.max(axis=1),
        "rms": np.sqrt(np.mean(x ** 2, axis=1)),
    }


__all__ = ["probe_time_series", "probe_statistics"]
