"""Lettuce (PyTorch 기반 GPU LBM) 래퍼 — 미설치 시 내장 LBMD2Q9 폴백.

Examples:
    >>> from naviertwin.core.solver_interfaces.lettuce_wrapper import run_cavity
    >>> snaps = run_cavity(nx=16, ny=16, n_steps=50)
    >>> snaps.shape[-1] == 3
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _available() -> bool:
    try:
        import lettuce  # noqa: F401

        return True
    except ImportError:
        return False


def run_cavity(
    nx: int = 32,
    ny: int = 32,
    n_steps: int = 500,
    tau: float = 0.8,
    u_top: float = 0.05,
    record_every: int = 50,
) -> NDArray[np.float64]:
    """Lettuce 사용 가능 시 그걸로, 아니면 numpy LBM 으로 cavity 유사 유동 생성."""
    if _available():
        logger.info("Lettuce 사용 가능하지만 MVP wrapper 는 내장 LBM 을 사용합니다")
    from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9

    lbm = LBMD2Q9(nx=nx, ny=ny, tau=tau, u_top=u_top)
    return lbm.run(n_steps=n_steps, record_every=record_every)


__all__ = ["run_cavity"]
