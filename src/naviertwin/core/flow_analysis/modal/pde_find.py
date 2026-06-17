"""PDE-FIND — 시공간 데이터에서 편미분 방정식 자동 복원.

Rudy et al., Science Advances 2017.

데이터 U[t, x] 로부터 다음을 구성:
    - Θ: {U, U², U_x, U U_x, U_xx, U_xx U, ...} (후보 항)
    - U_t
그리고 STRidge (sequentially-thresholded ridge) 로 희소 계수 추정.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.flow_analysis.modal.pde_find import pde_find_1d
    >>> from naviertwin.core.solver_interfaces.pde_solvers import solve_heat_1d
    >>> x = np.linspace(0, 1, 33)
    >>> u0 = np.sin(np.pi * x)
    >>> t, U = solve_heat_1d(u0, alpha=0.05, L=1.0, T=0.5, n_steps=500)
    >>> res = pde_find_1d(U, t, x, threshold=0.01)
    >>> "U_xx" in res["equation"]  # heat: u_t = α u_xx
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _derivative(
    U: NDArray[np.float64], axis: int, d: float, order: int = 1
) -> NDArray[np.float64]:
    """중앙차분 n차 도함수."""
    if _kernels is None:
        raise ImportError("naviertwin._native._kernels is required by PDE-FIND")
    return _kernels.derivative_2d(np.asarray(U, dtype=np.float64), d, axis, order)


def _build_library(
    U: NDArray[np.float64], dx: float
) -> tuple[NDArray[np.float64], list[str]]:
    """Θ 행렬 + 항 이름. U shape = (T, N)."""
    T, N = U.shape
    Ux = _derivative(U, axis=1, d=dx, order=1)
    Uxx = _derivative(U, axis=1, d=dx, order=2)

    def flat(A):
        return A.ravel()
    cols = {
        "1": np.ones(T * N),
        "U": flat(U),
        "U^2": flat(U ** 2),
        "U_x": flat(Ux),
        "U*U_x": flat(U * Ux),
        "U_xx": flat(Uxx),
        "U*U_xx": flat(U * Uxx),
    }
    names = list(cols.keys())
    theta_cols = []
    idx = 0
    while idx < len(names):
        theta_cols.append(cols[names[idx]])
        idx += 1
    Theta = np.stack(theta_cols, axis=1)
    return Theta, names


def _stridge(
    Theta: NDArray[np.float64], ut: NDArray[np.float64], threshold: float, max_iter: int = 10
) -> NDArray[np.float64]:
    coef, *_ = np.linalg.lstsq(Theta, ut, rcond=None)
    it = 0
    while it < max_iter:
        small = np.abs(coef) < threshold
        coef[small] = 0
        big_cols = np.where(~small)[0]
        if big_cols.size == 0:
            break
        coef_new, *_ = np.linalg.lstsq(Theta[:, big_cols], ut, rcond=None)
        coef = np.zeros_like(coef)
        coef[big_cols] = coef_new
        it += 1
    return coef


def pde_find_1d(
    U: NDArray[np.float64],
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    threshold: float = 0.01,
) -> dict[str, object]:
    """1D PDE discovery.

    Args:
        U: (T, N) 시공간 데이터.
        t: (T,) 시간 좌표 (등간격 가정).
        x: (N,) 공간 좌표 (등간격 가정).
        threshold: STRidge 희소화 임계.

    Returns:
        dict: {"coef", "names", "equation"}.
    """
    U = np.asarray(U, dtype=np.float64)
    if U.ndim != 2:
        raise ValueError("U (T, N) 2D 필요")
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    dt = float(t[1] - t[0])
    dx = float(x[1] - x[0])

    Ut = _derivative(U, axis=0, d=dt, order=1)
    Theta, names = _build_library(U, dx)
    coef = _stridge(Theta, Ut.ravel(), threshold=threshold)

    terms = []
    i = 0
    while i < len(names):
        if abs(coef[i]) > 0:
            terms.append(f"{coef[i]:.4g}*{names[i]}")
        i += 1
    equation = "U_t = " + (" + ".join(terms) if terms else "0")
    logger.info("PDE-FIND: %s", equation)
    return {"coef": coef, "names": names, "equation": equation}


__all__ = ["pde_find_1d"]
