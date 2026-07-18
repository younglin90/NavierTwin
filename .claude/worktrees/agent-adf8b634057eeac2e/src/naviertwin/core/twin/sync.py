"""Twin state synchronizer — pull source state, blend with twin (α).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.twin.sync import sync_state
    >>> twin = np.array([1.0, 2.0])
    >>> source = np.array([1.5, 2.5])
    >>> sync_state(twin, source, alpha=0.5)
    array([1.25, 2.25])
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sync_state(
    twin: NDArray[np.float64], source: NDArray[np.float64], *,
    alpha: float = 0.1,
) -> NDArray[np.float64]:
    return (1 - alpha) * np.asarray(twin) + alpha * np.asarray(source)


__all__ = ["sync_state"]
