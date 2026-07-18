"""GNAT (Gauss-Newton with Approximated Tensors) — Carlberg et al. 2013.

LSPG + sample-mesh DEIM 방식. 핵심: residual basis Φ_R, sample indices via DEIM.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.gnat import gnat_solve
    >>> A = np.diag([3.0, 2.0, 1.0, 0.5])
    >>> b = np.array([3.0, 2.0, 1.0, 0.5])
    >>> Phi = np.eye(4)[:, :3]
    >>> x = gnat_solve(A, b, Phi)
    >>> x.shape
    (4,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.dimensionality_reduction.linear.deim import deim


def gnat_solve(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    Phi: NDArray[np.float64],
    *,
    n_samples: int | None = None,
) -> NDArray[np.float64]:
    """GNAT: residual sampling via DEIM, LSPG over sampled rows."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    Phi = np.asarray(Phi, dtype=np.float64)
    k = Phi.shape[1]
    n_samples = n_samples or k
    # residual basis = A Φ (LSPG test space)
    Psi = A @ Phi
    # DEIM on Psi → sample indices
    Q, _ = np.linalg.qr(Psi)
    Q = Q[:, : min(Psi.shape[1], n_samples)]
    _, idx = deim(Q)
    # solve (Ψ_S)ᵀ Ψ_S ẑ = (Ψ_S)ᵀ b_S
    Psi_S = Psi[idx, :]
    b_S = b[idx]
    z, *_ = np.linalg.lstsq(Psi_S, b_S, rcond=None)
    return Phi @ z


__all__ = ["gnat_solve"]
