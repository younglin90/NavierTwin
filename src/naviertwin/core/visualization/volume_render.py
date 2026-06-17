"""Volume ray-marcher (CPU) — front-to-back compositing.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.visualization.volume_render import ray_march
    >>> vol = np.zeros((20, 20, 20))
    >>> vol[8:12, 8:12, 8:12] = 1.0
    >>> img = ray_march(vol, n_steps=20, axis=2)
    >>> img.shape
    (20, 20)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def ray_march(
    volume: NDArray[np.float64],
    *,
    n_steps: int = 32, axis: int = 2,
    alpha: float = 0.1,
) -> NDArray[np.float64]:
    """Front-to-back along `axis`. Returns 2D projection."""
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by ray_march")
    return _kernels.ray_march(np.asarray(volume, dtype=np.float64), n_steps, axis, alpha)


__all__ = ["ray_march"]
