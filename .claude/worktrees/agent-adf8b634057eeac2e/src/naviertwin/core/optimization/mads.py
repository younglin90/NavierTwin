"""MADS — Mesh Adaptive Direct Search (Audet & Dennis 2006), 간단 버전.

Poll 단계: 정직교 +/-e_i 방향 + 메쉬 크기 적응.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.mads import mads_minimize
    >>> x = mads_minimize(lambda x: float(x @ x), x0=np.array([1.0, 1.0]))
    >>> np.linalg.norm(x) < 0.05
    True
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def mads_minimize(
    f: Callable[[NDArray], float],
    x0: NDArray[np.float64],
    *,
    delta_init: float = 1.0,
    delta_min: float = 1e-4,
    max_iter: int = 200,
) -> NDArray[np.float64]:
    x = np.asarray(x0, dtype=np.float64).copy()
    n = x.shape[0]
    delta = delta_init
    fx = f(x)
    iter_count = 0
    while delta > delta_min and iter_count < max_iter:
        # poll: ±e_i
        success = False
        i = 0
        signs = (+1, -1)
        while i < n:
            sign_idx = 0
            while sign_idx < len(signs):
                trial = x.copy()
                trial[i] += signs[sign_idx] * delta
                ft = f(trial)
                if ft < fx:
                    x = trial
                    fx = ft
                    success = True
                    break
                sign_idx += 1
            if success:
                break
            i += 1
        if not success:
            delta *= 0.5
        else:
            delta *= 1.0  # keep
        iter_count += 1
    return x


__all__ = ["mads_minimize"]
