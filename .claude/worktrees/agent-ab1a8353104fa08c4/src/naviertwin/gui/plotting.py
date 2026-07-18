"""간단한 matplotlib 플롯 헬퍼 — non-GUI (Agg) 안전.

Examples:
    >>> import numpy as np
    >>> from naviertwin.gui.plotting import plot_field_2d  # doctest: +SKIP
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _ensure_mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise RuntimeError("matplotlib 필요") from exc


def plot_field_2d(
    field: NDArray[np.float64],
    path: str | Path,
    *,
    title: str = "",
    cmap: str = "viridis",
    vmin: float | None = None, vmax: float | None = None,
    colorbar: bool = True,
) -> Path:
    plt = _ensure_mpl()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        np.asarray(field).T, origin="lower", cmap=cmap,
        vmin=vmin, vmax=vmax, aspect="auto",
    )
    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return p


def plot_line(
    x: NDArray, y: NDArray, path: str | Path,
    *, title: str = "", xlabel: str = "", ylabel: str = "",
) -> Path:
    plt = _ensure_mpl()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return p


def plot_loss_curve(
    history: list[float], path: str | Path,
    *, title: str = "loss",
) -> Path:
    plt = _ensure_mpl()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(history)
    ax.set_yscale("log")
    ax.set_xlabel("iter")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return p


def make_figure(nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] = (6, 4)) -> Any:
    plt = _ensure_mpl()
    return plt.subplots(nrows, ncols, figsize=figsize)


__all__ = ["plot_field_2d", "plot_line", "plot_loss_curve", "make_figure"]
