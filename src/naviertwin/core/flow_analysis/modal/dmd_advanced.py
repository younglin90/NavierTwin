"""PyDMD 고급 variants 래퍼 — HODMD / MrDMD / OptDMD / HAVOK / DMDc.

기존 `dmd.py` 는 기본 DMD 만. 이 모듈은 다양한 DMD variant 를 통합 API 로.

- HODMD: High-order DMD — noisy 시계열 강건
- MrDMD: Multi-resolution DMD — 시간 스케일 분리
- OptDMD: Optimized DMD — 모든 스냅샷 글로벌 최적화
- HAVOK: Hankel Alternative View Of Koopman
- DMDc: DMD with control inputs

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.flow_analysis.modal.dmd_advanced import (
    ...     hodmd_analysis,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 100))
    >>> result = hodmd_analysis(X, svd_rank=5, d=3)
    >>> "modes" in result and "eigenvalues" in result
    True
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _require_pydmd() -> None:
    try:
        import pydmd  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pydmd 필요: pip install pydmd") from exc


def hodmd_analysis(
    X: NDArray[np.float64],
    svd_rank: int = 0,
    d: int = 2,
) -> dict[str, Any]:
    """High-order DMD — 시간 지연 d 만큼 증강."""
    _require_pydmd()
    from pydmd import HODMD

    dmd = HODMD(svd_rank=svd_rank, d=d).fit(X)
    return {
        "modes": np.asarray(dmd.modes),
        "eigenvalues": np.asarray(dmd.eigs),
        "amplitudes": np.asarray(dmd.amplitudes),
        "dmd_object": dmd,
    }


def mrdmd_analysis(
    X: NDArray[np.float64],
    max_level: int = 4,
    max_cycles: int = 1,
) -> dict[str, Any]:
    """Multi-resolution DMD — 시간 스케일 분리."""
    _require_pydmd()
    from pydmd import DMD, MrDMD

    inner = DMD()
    dmd = MrDMD(dmd=inner, max_level=max_level, max_cycles=max_cycles).fit(X)
    modes_by_level = list(
        map(lambda lvl: np.asarray(dmd.partial_modes(lvl)), range(max_level))
    )
    return {
        "dmd_object": dmd,
        "modes_by_level": modes_by_level,
    }


def optdmd_analysis(
    X: NDArray[np.float64],
    svd_rank: int = 0,
    factorization: str = "evd",
) -> dict[str, Any]:
    """OptDMD — 전체 스냅샷 글로벌 최적화."""
    _require_pydmd()
    from pydmd import OptDMD

    dmd = OptDMD(svd_rank=svd_rank, factorization=factorization).fit(X)
    return {
        "modes": np.asarray(dmd.modes),
        "eigenvalues": np.asarray(dmd.eigs),
        "amplitudes": np.asarray(dmd.amplitudes),
        "dmd_object": dmd,
    }


def havok_analysis(
    X: NDArray[np.float64],
    delays: int = 10,
    svd_rank: int = 5,
) -> dict[str, Any]:
    """HAVOK — Hankel 기반 forced linear system."""
    _require_pydmd()
    from pydmd import HAVOK

    # HAVOK 은 1D 시계열을 받음
    if X.ndim == 2 and X.shape[0] == 1:
        x = X.ravel()
    elif X.ndim == 1:
        x = X
    else:
        # 첫 번째 채널 사용
        x = X[0]
    dmd = HAVOK(svd_rank=svd_rank, delays=delays).fit(x)
    return {
        "A": np.asarray(dmd.A),
        "eigenvalues": np.asarray(dmd.eigs) if hasattr(dmd, "eigs") else None,
        "dmd_object": dmd,
    }


def dmdc_analysis(
    X: NDArray[np.float64],
    U: NDArray[np.float64],
    svd_rank: int = 0,
) -> dict[str, Any]:
    """DMD with control inputs — x_{k+1} = A x_k + B u_k."""
    _require_pydmd()
    from pydmd import DMDc

    dmd = DMDc(svd_rank=svd_rank)
    dmd.fit(X, U)
    return {
        "A": np.asarray(dmd.A) if hasattr(dmd, "A") else None,
        "B": np.asarray(dmd.B) if hasattr(dmd, "B") else None,
        "modes": np.asarray(dmd.modes),
        "eigenvalues": np.asarray(dmd.eigs),
        "dmd_object": dmd,
    }


__all__ = [
    "hodmd_analysis",
    "mrdmd_analysis",
    "optdmd_analysis",
    "havok_analysis",
    "dmdc_analysis",
]
