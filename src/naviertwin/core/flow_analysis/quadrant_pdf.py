"""사분면 분석 (Quadrant Analysis) + 확률밀도 함수 (PDF) 추정.

Tecplot 360, EnSight, MATLAB의 표준 난류 통계 분석 기능. u'v' 신호를
Q1-Q4 사상으로 분류하고 (이젝션, 스윕 식별), KDE/히스토그램 PDF 산출.

References:
    Wallace, J.M. et al., "The wall region in turbulent shear flow",
    JFM 54:39-48, 1972 (quadrant analysis 정의).

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> up = rng.standard_normal(1000)
    >>> vp = rng.standard_normal(1000)
    >>> from naviertwin.core.flow_analysis.quadrant_pdf import quadrant_split
    >>> q = quadrant_split(up, vp, hole=1.0)
    >>> set(q.keys()) >= {"Q1", "Q2", "Q3", "Q4"}
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")

logger = get_logger(__name__)


def quadrant_split(
    up: NDArray[np.float64],
    vp: NDArray[np.float64],
    hole: float = 0.0,
) -> dict[str, dict[str, float | int]]:
    """u'v' 신호를 Q1-Q4 사분면으로 분류 + 각 영역 통계 산출.

    Quadrants:
        Q1 (u'>0, v'>0) — 외향 가속 (outward interaction)
        Q2 (u'<0, v'>0) — 이젝션 (ejection, 가장 중요)
        Q3 (u'<0, v'<0) — 내향 감속 (inward interaction)
        Q4 (u'>0, v'<0) — 스윕 (sweep, 두 번째로 중요)

    Hole region (|u'v'| < H · u_rms · v_rms): 약한 이벤트는 제외.

    Args:
        up: u 변동 신호.
        vp: v 변동 신호.
        hole: hole 크기 H ≥ 0.

    Returns:
        {"Q1": {...}, "Q2": {...}, ...} 각 사분면의 (count, fraction, mean_uv).

    Raises:
        ValueError: 형상 불일치 또는 hole < 0.
    """
    up = np.asarray(up, dtype=np.float64).ravel()
    vp = np.asarray(vp, dtype=np.float64).ravel()
    if up.shape != vp.shape:
        raise ValueError(f"shape mismatch: {up.shape} vs {vp.shape}")
    if hole < 0:
        raise ValueError(f"hole must be >= 0, got {hole}")

    return dict(_kernels.quadrant_split(up, vp, float(hole)))


def histogram_pdf(
    x: NDArray[np.float64],
    bins: int = 50,
    range_: tuple[float, float] | None = None,
    normalize: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """히스토그램 기반 PDF 추정.

    Args:
        x: (N,) 데이터.
        bins: 빈 수.
        range_: (low, high) 빈 경계. None이면 자동.
        normalize: True면 PDF (적분 = 1), False면 빈도 카운트.

    Returns:
        (centers, pdf): 빈 중심값과 PDF.

    Raises:
        ValueError: bins ≤ 0 또는 x가 1D 아님.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if bins <= 0:
        raise ValueError(f"bins must be > 0, got {bins}")
    counts, edges = np.histogram(x, bins=bins, range=range_)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if normalize:
        widths = np.diff(edges)
        pdf = counts / (counts.sum() * widths + 1e-30)
        return centers, pdf
    return centers, counts.astype(np.float64)


def kde_pdf(
    x: NDArray[np.float64],
    points: NDArray[np.float64] | None = None,
    bandwidth: float | None = None,
    n_eval: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """가우시안 커널 밀도 추정 (KDE).

    Bandwidth가 None이면 Scott's rule: h = N^(-1/5) · σ.

    Args:
        x: (N,) 표본.
        points: 평가 점 (선택). None이면 [min, max] 범위에 n_eval 점.
        bandwidth: 커널 폭. None이면 Scott's rule.
        n_eval: 자동 평가 점 수.

    Returns:
        (eval_points, pdf).

    Raises:
        ValueError: x가 1D 아님 또는 표본 수 < 2.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    N = x.size
    if N < 2:
        raise ValueError(f"need at least 2 samples, got {N}")

    if bandwidth is None:
        sigma = max(x.std(), 1e-30)
        bandwidth = N ** (-1 / 5.0) * sigma
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be > 0, got {bandwidth}")

    if points is None:
        lo, hi = x.min() - 3 * bandwidth, x.max() + 3 * bandwidth
        points = np.linspace(lo, hi, n_eval)
    points = np.asarray(points, dtype=np.float64)

    # K(x) = (1/√(2π)) exp(-x²/2)
    diff = (points[:, None] - x[None, :]) / bandwidth
    kvals = np.exp(-0.5 * diff ** 2) / np.sqrt(2.0 * np.pi)
    pdf = kvals.mean(axis=1) / bandwidth
    return points, pdf


def joint_pdf_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    bins: int = 50,
    range_: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """2D 결합 확률 밀도 추정 (히스토그램 기반).

    Args:
        x, y: 같은 길이의 1D 신호.
        bins: 각 축 빈 수.
        range_: ((x_lo, x_hi), (y_lo, y_hi)). None이면 자동.

    Returns:
        (x_centers, y_centers, pdf_2d) — pdf_2d 형상 (bins, bins).

    Raises:
        ValueError: 형상 불일치.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range_)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    dx = np.diff(xedges)
    dy = np.diff(yedges)
    area = dx[:, None] * dy[None, :]
    pdf = H / (H.sum() * area + 1e-30)
    return xc, yc, pdf


__all__ = ["quadrant_split", "histogram_pdf", "kde_pdf", "joint_pdf_2d"]
