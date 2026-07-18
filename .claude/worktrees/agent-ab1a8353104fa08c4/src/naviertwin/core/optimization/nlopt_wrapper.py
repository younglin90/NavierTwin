"""NLopt 30+ 알고리즘 통합 래퍼.

NLopt 는 C 기반 고성능 비선형 최적화 라이브러리로, 다양한 gradient-free
및 gradient-based 알고리즘을 제공한다.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.nlopt_wrapper import nlopt_minimize
    >>>
    >>> def obj(x):
    ...     return float((x[0] - 1.5) ** 2 + (x[1] + 2.0) ** 2)
    >>>
    >>> x_best, f_best = nlopt_minimize(
    ...     obj, x0=np.array([0.0, 0.0]),
    ...     bounds=np.array([[-5, 5], [-5, 5]]),
    ...     algorithm="LN_BOBYQA", max_eval=200,
    ... )
    >>> f_best < 0.01
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


# 주요 알고리즘 카탈로그 (prefix: G=global, L=local / N=no-gradient, D=gradient)
ALGORITHMS = {
    # Gradient-free local
    "LN_COBYLA": "constraint-aware",
    "LN_BOBYQA": "bound-quadratic",
    "LN_NELDERMEAD": "simplex",
    "LN_SBPLX": "subplex",
    "LN_PRAXIS": "Brent principal axis",
    # Gradient local
    "LD_LBFGS": "L-BFGS",
    "LD_MMA": "Method of Moving Asymptotes",
    "LD_SLSQP": "Sequential LSQ programming",
    # Global
    "GN_DIRECT": "DIRECT",
    "GN_DIRECT_L": "DIRECT local bias",
    "GN_CRS2_LM": "Controlled Random Search 2",
    "GN_ISRES": "Improved Stochastic Ranking ES",
    "GN_ESCH": "Evolutionary",
    "GN_AGS": "AGS global",
}


def nlopt_minimize(
    f: Callable[[NDArray[np.float64]], float],
    x0: NDArray[np.float64],
    bounds: NDArray[np.float64],
    algorithm: str = "LN_BOBYQA",
    max_eval: int = 500,
    xtol_rel: float = 1e-6,
    grad_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> tuple[NDArray[np.float64], float]:
    """NLopt 로 f 를 최소화한다.

    Args:
        f: 목적함수.
        x0: 초기점.
        bounds: (n, 2).
        algorithm: ALGORITHMS 키.
        max_eval: 최대 평가 수.
        xtol_rel: 상대 수렴 기준.
        grad_fn: gradient-based 알고리즘 필요 시 제공.

    Returns:
        (x_best, f_best).
    """
    try:
        import nlopt
    except ImportError as exc:
        raise RuntimeError("nlopt 필요: pip install nlopt") from exc

    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"알 수 없는 algorithm '{algorithm}'. 가능: {list(ALGORITHMS)}"
        )

    x0 = np.asarray(x0, dtype=np.float64).ravel()
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[0] != x0.size:
        raise ValueError("bounds shape 불일치")

    opt = nlopt.opt(getattr(nlopt, algorithm), x0.size)
    opt.set_lower_bounds(bounds[:, 0].tolist())
    opt.set_upper_bounds(bounds[:, 1].tolist())
    opt.set_maxeval(max_eval)
    opt.set_xtol_rel(xtol_rel)

    def nlopt_obj(x: np.ndarray, grad: np.ndarray) -> float:
        val = float(f(np.asarray(x)))
        if grad.size > 0 and grad_fn is not None:
            g = np.asarray(grad_fn(x), dtype=np.float64).ravel()
            if g.size == grad.size:
                grad[:] = g
        return val

    opt.set_min_objective(nlopt_obj)
    try:
        x_best = opt.optimize(x0)
        f_best = float(opt.last_optimum_value())
    except Exception as e:  # noqa: BLE001
        logger.warning("NLopt 경고: %s", e)
        x_best = x0
        f_best = float(f(x0))

    logger.info(
        "NLopt %s 완료: f_best=%.6g, n_eval=%d",
        algorithm, f_best, opt.get_numevals(),
    )
    return np.asarray(x_best, dtype=np.float64), f_best


def list_algorithms() -> dict[str, str]:
    """사용 가능한 알고리즘 이름 → 설명."""
    return dict(ALGORITHMS)


__all__ = ["nlopt_minimize", "list_algorithms", "ALGORITHMS"]
