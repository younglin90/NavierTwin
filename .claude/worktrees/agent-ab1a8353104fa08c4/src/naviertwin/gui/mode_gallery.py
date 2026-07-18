"""POD mode gallery — 각 mode 의 이미지를 grid 로 저장.

Examples:
    >>> # test 참조
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def render_mode_gallery(
    modes: NDArray[np.float64], shape: tuple[int, int],
    path: str | Path,
    *, cmap: str = "viridis", cols: int = 4, title_prefix: str = "Mode",
) -> Path:
    """modes: (n_features, n_modes), shape: (nx, ny). PNG grid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib 필요") from exc

    modes = np.asarray(modes, dtype=np.float64)
    n_modes = modes.shape[1]
    rows = int(np.ceil(n_modes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes_arr: Any = np.atleast_2d(axes).ravel()
    i = 0
    while i < n_modes:
        ax = axes_arr[i]
        img = modes[:, i].reshape(shape)
        ax.imshow(img.T, origin="lower", cmap=cmap, aspect="auto")
        ax.set_title(f"{title_prefix} {i + 1}")
        ax.axis("off")
        i += 1
    j = n_modes
    while j < rows * cols:
        axes_arr[j].axis("off")
        j += 1
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return p


__all__ = ["render_mode_gallery"]
