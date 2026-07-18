"""Hybrid ROM 고도화 — POD-Galerkin + NN 잔차 + 물리 제약 투영.

HybridROM 에 reconstruct 후 선형 제약 투영 단계를 추가.
    x_rec = U a + NN(a)        (기본 HybridROM)
    x_proj = Π(x_rec)            (추가 보정)

여기서 Π 는 C x = d 를 만족하는 null-space 투영.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD
    >>> from naviertwin.core.physics_correction.hybrid_rom_adv import HybridROMAdv
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((40, 20))
    >>> X = X - X.mean(axis=0, keepdims=True)  # 행합 0
    >>> pod = SnapshotPOD(n_modes=3); pod.fit(X)
    >>> C = np.ones((1, 40))  # 행합 = 0 제약
    >>> d = np.zeros(1)
    >>> m = HybridROMAdv(reducer=pod, C=C, d=d, hidden=16, max_epochs=20)
    >>> m.fit(X)
    >>> X_rec = m.reconstruct(X)
    >>> float(np.abs((C @ X_rec - d[:, None]).max())) < 1e-8
    True
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.physics_correction.hybrid_rom import HybridROM
from naviertwin.core.physics_correction.physics_correction import (
    project_linear_constraint,
)
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class HybridROMAdv(HybridROM):
    """HybridROM + 선형 제약 사후 투영."""

    def __init__(
        self,
        reducer: Any,
        C: NDArray[np.float64] | None = None,
        d: NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(reducer=reducer, **kwargs)
        if (C is None) != (d is None):
            raise ValueError("C 와 d 는 함께 지정해야 합니다")
        self.C = None if C is None else np.asarray(C, dtype=np.float64)
        self.d = None if d is None else np.asarray(d, dtype=np.float64)

    def reconstruct(self, snapshots: NDArray[np.float64]) -> NDArray[np.float64]:
        X_rec = super().reconstruct(snapshots)
        if self.C is not None and self.d is not None:
            # X_rec: (n_features, n_snap) → 열별 투영
            rec_T = X_rec.T  # (n_snap, n_features)
            rec_T = project_linear_constraint(rec_T, self.C, self.d)
            X_rec = rec_T.T
            logger.debug(
                "HybridROMAdv 제약 투영 적용: max|Cx-d|=%.4g",
                float(np.abs(self.C @ X_rec - self.d[:, None]).max()),
            )
        return X_rec


__all__ = ["HybridROMAdv"]
