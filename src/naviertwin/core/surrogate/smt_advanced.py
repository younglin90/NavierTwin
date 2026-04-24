"""SMT 고급 surrogate + DOE 래퍼.

기존 `rbf_surrogate.py` / `kriging_surrogate.py` 외에:
    - KPLS: 고차원 Kriging Partial Least Squares
    - GEKPLS: Gradient-Enhanced KPLS (gradient 정보 활용)
    - IDW: Inverse Distance Weighting
    - QP: Quadratic Polynomial
    - DOE: Latin Hypercube, FullFactorial, Random

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.surrogate.smt_advanced import (
    ...     kpls_fit, lhs_design, full_factorial,
    ... )
    >>> X = lhs_design(n=30, d=2, xlimits=np.array([[-1, 1], [-1, 1]]), seed=0)
    >>> X.shape
    (30, 2)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _require_smt() -> None:
    try:
        import smt  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("smt 필요: pip install smt") from exc


def lhs_design(
    n: int, d: int, xlimits: NDArray[np.float64], criterion: str = "ese",
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Latin Hypercube Sampling (SMT 기반)."""
    _require_smt()
    from smt.sampling_methods import LHS

    kw = {"xlimits": np.asarray(xlimits, dtype=np.float64), "criterion": criterion}
    try:
        lhs = LHS(**kw, random_state=seed)
    except (TypeError, AssertionError):
        # 일부 SMT 버전은 random_state 미지원
        if seed is not None:
            np.random.seed(seed)
        lhs = LHS(**kw)
    return np.asarray(lhs(n), dtype=np.float64)


def full_factorial(
    n_per_axis: int, xlimits: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Full-factorial grid."""
    _require_smt()
    from smt.sampling_methods import FullFactorial

    ff = FullFactorial(xlimits=np.asarray(xlimits, dtype=np.float64))
    return np.asarray(ff(n_per_axis ** xlimits.shape[0]), dtype=np.float64)


def kpls_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    n_comp: int = 2,
) -> Any:
    """KPLS — 고차원 대비 차원축소 결합 Kriging."""
    _require_smt()
    from smt.surrogate_models import KPLS

    model = KPLS(n_comp=n_comp, print_global=False)
    model.set_training_values(
        np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    model.train()
    logger.info("KPLS fit: n_comp=%d, n=%d", n_comp, X.shape[0])
    return model


def gekpls_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    grads: NDArray[np.float64],
    n_comp: int = 2,
    extra_points: int = 2,
) -> Any:
    """Gradient-Enhanced KPLS — gradient 정보 사용."""
    _require_smt()
    from smt.surrogate_models import GEKPLS

    model = GEKPLS(n_comp=n_comp, extra_points=extra_points, print_global=False)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    grads = np.asarray(grads, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    model.set_training_values(X, y)
    for j in range(X.shape[1]):
        model.set_training_derivatives(X, grads[:, j : j + 1], j)
    model.train()
    logger.info("GEKPLS fit: n=%d, d=%d", X.shape[0], X.shape[1])
    return model


def idw_fit(X: NDArray[np.float64], y: NDArray[np.float64], p: float = 2.0) -> Any:
    """Inverse Distance Weighting."""
    _require_smt()
    from smt.surrogate_models import IDW

    model = IDW(p=p, print_global=False)
    model.set_training_values(
        np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    model.train()
    return model


def qp_fit(X: NDArray[np.float64], y: NDArray[np.float64]) -> Any:
    """Quadratic Polynomial surrogate."""
    _require_smt()
    from smt.surrogate_models import QP

    model = QP(print_global=False)
    model.set_training_values(
        np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    model.train()
    return model


def smt_predict(model: Any, X: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(model.predict_values(np.asarray(X, dtype=np.float64)), dtype=np.float64)


__all__ = [
    "lhs_design",
    "full_factorial",
    "kpls_fit",
    "gekpls_fit",
    "idw_fit",
    "qp_fit",
    "smt_predict",
]
