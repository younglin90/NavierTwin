"""Proper Generalized Decomposition (PGD) — alternating rank-1 분해.

다파라미터 텐서 X[i,j,k] ≈ Σ_m F_m(i) G_m(j) H_m(k) 를 탐욕적으로 추가.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.flow_analysis.modal.pgd import compute_pgd_3d
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((10, 8, 6))
    >>> modes = compute_pgd_3d(X, n_modes=3, max_iter=50)
    >>> len(modes)
    3
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _power_iter(X: NDArray[np.float64], n_iter: int, tol: float) -> tuple:
    """3D 텐서의 rank-1 근사."""
    nI, J, K = X.shape
    rng = np.random.default_rng(0)
    F = rng.standard_normal(nI)
    G = rng.standard_normal(J)
    H = rng.standard_normal(K)
    prev_norm = np.inf
    it = 0
    while it < n_iter:
        # mode-1: F
        F = np.einsum("ijk,j,k->i", X, G, H)
        nF = np.linalg.norm(F)
        if nF < 1e-30:
            break
        F /= nF
        # mode-2: G
        G = np.einsum("ijk,i,k->j", X, F, H)
        nG = np.linalg.norm(G)
        if nG < 1e-30:
            break
        G /= nG
        # mode-3: H
        H = np.einsum("ijk,i,j->k", X, F, G)
        nH = np.linalg.norm(H)
        if nH < 1e-30:
            break
        H /= nH

        total = nF * nG * nH
        if abs(prev_norm - total) < tol:
            break
        prev_norm = total
        it += 1
    sigma = float(np.einsum("ijk,i,j,k->", X, F, G, H))
    return F, G, H, sigma


def compute_pgd_3d(
    X: NDArray[np.float64],
    n_modes: int = 5,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> list[dict[str, NDArray[np.float64] | float]]:
    """3D 텐서의 greedy PGD 분해.

    Returns:
        각 mode: {"F", "G", "H", "sigma"}. X ≈ Σ sigma · F⊗G⊗H.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"3D 텐서 필요: {X.shape}")

    R = X.copy()
    modes: list[dict] = []
    mode_idx = 0
    while mode_idx < n_modes:
        F, G, H, sigma = _power_iter(R, max_iter, tol)
        modes.append({"F": F, "G": G, "H": H, "sigma": sigma})
        R = R - sigma * np.einsum("i,j,k->ijk", F, G, H)
        mode_idx += 1
    logger.info(
        "PGD 3D 완료: n_modes=%d, residual=%.4g",
        n_modes,
        float(np.linalg.norm(R)),
    )
    return modes


def reconstruct_pgd(
    modes: list[dict[str, NDArray[np.float64] | float]],
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """PGD 모드로부터 원본 텐서 복원."""
    if not modes:
        return np.zeros(shape, dtype=np.float64)
    parts = tuple(
        map(
            lambda m: m["sigma"] * np.einsum("i,j,k->ijk", m["F"], m["G"], m["H"]),
            modes,
        )
    )
    return np.sum(np.stack(parts), axis=0)


__all__ = ["compute_pgd_3d", "reconstruct_pgd"]
