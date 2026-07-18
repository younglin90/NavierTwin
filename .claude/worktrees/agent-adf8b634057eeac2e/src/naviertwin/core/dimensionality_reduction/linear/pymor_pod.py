"""pyMOR 래퍼 — POD / DEIM / Gram-Schmidt orthonormalization.

pyMOR 는 검증된 ROM 라이브러리. VectorArray 추상화 위에서 BPOD/POD-Galerkin/
DEIM/EIM 등을 안정적으로 제공.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.pymor_pod import (
    ...     pymor_pod, pymor_deim,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 20))
    >>> modes, svals = pymor_pod(X, modes=5)
    >>> modes.shape
    (50, 5)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _require_pymor() -> None:
    try:
        import pymor  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pymor 필요: pip install pymor") from exc


def _to_va(snapshots: NDArray[np.float64]):
    """(n_feat, n_snap) → pyMOR VectorArray (n_snap vectors, each len n_feat)."""
    from pymor.vectorarrays.numpy import NumpyVectorSpace

    arr = np.asarray(snapshots, dtype=np.float64)
    sp = NumpyVectorSpace(arr.shape[0])
    # pyMOR from_numpy: (dim, n_vectors). 우리 (n_feat, n_snap) 그대로.
    return sp.from_numpy(arr.copy())


def pymor_pod(
    snapshots: NDArray[np.float64],
    modes: int = 10,
    rtol: float = 1e-7,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """pyMOR POD — VectorArray → basis.

    Args:
        snapshots: (n_feat, n_snap).
        modes: 최대 모드 수.
        rtol: 상대 절단 오차.

    Returns:
        (POD modes (n_feat, k), singular values (k,)).
    """
    _require_pymor()
    from pymor.algorithms.pod import pod

    va = _to_va(np.asarray(snapshots, dtype=np.float64))
    basis, svals = pod(va, modes=modes, rtol=rtol)
    modes_np = np.asarray(basis.to_numpy(), dtype=np.float64)
    if modes_np.shape[0] != snapshots.shape[0]:
        modes_np = modes_np.T
    svals_np = np.asarray(svals, dtype=np.float64)
    logger.info("pyMOR POD: %d 모드, top σ=%.4g", modes_np.shape[1], svals_np[0] if svals_np.size else 0)
    return modes_np, svals_np


def pymor_deim(
    snapshots: NDArray[np.float64],
    modes: int = 10,
    product: object | None = None,
) -> dict[str, NDArray[np.float64] | np.ndarray]:
    """pyMOR DEIM — 비선형 항 hyper-reduction.

    Returns:
        dict(interpolation_matrix, interpolation_indices, collateral_basis).
    """
    _require_pymor()
    from pymor.algorithms.ei import deim

    va = _to_va(np.asarray(snapshots, dtype=np.float64))
    result = deim(va, modes=modes, product=product)
    # pyMOR 버전별 반환 arity 차이
    # pymor 최신: (indices, collateral_VectorArray, data_dict)
    if len(result) == 3:
        interp_idx, collateral_va, data = result
        interp_mat = data.get("matrix") if isinstance(data, dict) else None
        singular_values = (
            data.get("svals", np.array([])) if isinstance(data, dict) else np.array([])
        )
    elif len(result) == 4:
        interp_idx, interp_mat, collateral_va, singular_values = result
    else:
        raise RuntimeError(f"예기치 않은 deim 반환: {len(result)} 항목")
    col_arr = (
        np.asarray(collateral_va.to_numpy(), dtype=np.float64)
        if hasattr(collateral_va, "to_numpy")
        else np.asarray(collateral_va, dtype=np.float64)
    )
    if interp_mat is None:
        interp_mat = np.zeros((len(interp_idx), len(interp_idx)))
    else:
        interp_mat = np.asarray(interp_mat, dtype=np.float64)
    logger.info(
        "pyMOR DEIM: %d indices, basis shape=%s",
        len(interp_idx), col_arr.shape,
    )
    return {
        "interpolation_indices": np.asarray(interp_idx, dtype=np.int64),
        "interpolation_matrix": interp_mat,
        "collateral_basis": col_arr,
        "singular_values": np.asarray(singular_values, dtype=np.float64),
    }


def pymor_gram_schmidt(
    snapshots: NDArray[np.float64],
    atol: float = 1e-13,
    rtol: float = 1e-13,
) -> NDArray[np.float64]:
    """pyMOR numerically stable Gram-Schmidt orthonormalization.

    Returns:
        orthonormal basis (n_feat, n_ortho).
    """
    _require_pymor()
    from pymor.algorithms.gram_schmidt import gram_schmidt

    va = _to_va(np.asarray(snapshots, dtype=np.float64))
    ortho = gram_schmidt(va, atol=atol, rtol=rtol, copy=True)
    q = np.asarray(ortho.to_numpy(), dtype=np.float64)
    if q.shape[0] != snapshots.shape[0]:
        q = q.T
    return q


__all__ = ["pymor_pod", "pymor_deim", "pymor_gram_schmidt"]
