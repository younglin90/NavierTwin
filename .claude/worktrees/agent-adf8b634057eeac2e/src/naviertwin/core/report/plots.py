"""Publication-quality matplotlib 프리셋 + 편의 플롯 함수.

- apply_publication_style: TeX-like 폰트 + serif + 큰 tick/label
- plot_loss_curve, plot_compare_metrics, plot_field_2d, plot_pod_energy

저널/논문 제출 수준의 가독성 (linewidth, dpi, label size).

Examples:
    >>> from naviertwin.core.report.plots import apply_publication_style, plot_loss_curve
    >>> apply_publication_style()
    >>> fig = plot_loss_curve({"FNO": [1.0, 0.5, 0.1]}, log_scale=True)
    >>> fig is not None
    True
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def apply_publication_style(
    font_family: str = "serif",
    font_size: int = 12,
    dpi: int = 150,
) -> None:
    """Matplotlib rcParams 를 논문급으로 설정."""
    try:
        import matplotlib as mpl
    except ImportError as exc:
        raise RuntimeError("matplotlib 필요") from exc

    mpl.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 2,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.dpi": dpi,
        "savefig.dpi": dpi * 2,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.6,
        "axes.linewidth": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    })


def plot_loss_curve(
    losses: dict[str, list[float]],
    log_scale: bool = True,
    ax: Any = None,
) -> Any:
    """여러 계열 loss curve 비교 플롯."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.2))
    else:
        fig = ax.figure
    def plot_one(item: tuple[str, list[float]]) -> None:
        label, L = item
        if not L:
            return
        ax.plot(range(1, len(L) + 1), L, label=label)

    tuple(map(plot_one, losses.items()))
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if log_scale:
        ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return fig


def plot_compare_metrics(
    results: dict[str, dict[str, float]],
    metric: str = "rmse",
    ax: Any = None,
) -> Any:
    """여러 모델의 단일 metric 막대 그래프."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.2))
    else:
        fig = ax.figure
    names = list(results.keys())
    vals = list(map(lambda n: results[n].get(metric, 0.0), names))
    ax.bar(names, vals, color="#3B6FCC")
    ax.set_ylabel(metric)
    tuple(map(lambda lab: lab.set_rotation(30), ax.get_xticklabels()))
    return fig


def plot_field_2d(
    field: NDArray[np.float64],
    title: str = "",
    cmap: str = "RdBu_r",
    ax: Any = None,
) -> Any:
    """2D 스칼라 필드 컨투어맵."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4))
    else:
        fig = ax.figure
    arr = np.asarray(field, dtype=np.float64)
    im = ax.imshow(arr, cmap=cmap, origin="lower", aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    return fig


def plot_pod_energy(
    singular_values: NDArray[np.float64],
    ax: Any = None,
) -> Any:
    """POD 특이값 에너지 누적 곡선."""
    import matplotlib.pyplot as plt

    s = np.asarray(singular_values, dtype=np.float64)
    energy = s ** 2
    cum = np.cumsum(energy) / max(energy.sum(), 1e-30)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.2))
    else:
        fig = ax.figure
    ax.plot(range(1, len(cum) + 1), cum, marker="o", color="#C05858")
    ax.axhline(0.99, linestyle="--", color="gray", alpha=0.7, label="99%")
    ax.set_xlabel("mode index")
    ax.set_ylabel("cumulative energy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    return fig


__all__ = [
    "apply_publication_style",
    "plot_loss_curve",
    "plot_compare_metrics",
    "plot_field_2d",
    "plot_pod_energy",
]
