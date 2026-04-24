"""PDE 벤치마크 데이터셋 생성 레지스트리.

다양한 파라미터에 대해 자동으로 (파라미터, 스냅샷) 쌍을 생성한다.

Examples:
    >>> from naviertwin.core.benchmarks.dataset_catalog import (
    ...     generate_burgers_dataset, generate_heat_dataset,
    ... )
    >>> data = generate_burgers_dataset(n_samples=5, n_x=32)
    >>> data["params"].shape[0]
    5
    >>> data["snapshots"].shape
    (5, 32)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def generate_burgers_dataset(
    n_samples: int = 20,
    n_x: int = 64,
    T: float = 0.5,
    n_steps: int = 200,
    seed: int | None = 0,
) -> dict[str, NDArray[np.float64]]:
    """Burgers 데이터셋 — 파라미터: [nu, A_1, A_2, A_3].

    각 샘플마다 저주파 sin 조합 초기조건 생성 → T 시점의 스냅샷.
    """
    from naviertwin.core.solver_interfaces.pde_solvers import solve_burgers_1d

    rng = np.random.default_rng(seed)
    params = np.zeros((n_samples, 4))
    snaps = np.zeros((n_samples, n_x))
    x = np.linspace(0, 2 * np.pi, n_x, endpoint=False)

    for i in range(n_samples):
        nu = float(10 ** rng.uniform(-2.5, -1.5))  # 0.003 ~ 0.03
        amps = rng.uniform(-1, 1, 3)
        u0 = sum(amps[k] * np.sin((k + 1) * x) for k in range(3))
        _, U = solve_burgers_1d(u0, nu=nu, L=2 * np.pi, T=T, n_steps=n_steps)
        params[i] = [nu, *amps]
        snaps[i] = U[-1]

    logger.info("Burgers 데이터셋: %d samples, n_x=%d", n_samples, n_x)
    return {"params": params, "snapshots": snaps, "x": x}


def generate_heat_dataset(
    n_samples: int = 20,
    n_x: int = 33,
    T: float = 0.5,
    n_steps: int = 500,
    seed: int | None = 0,
) -> dict[str, NDArray[np.float64]]:
    """Heat 데이터셋 — 파라미터: [alpha, A_1, A_2]."""
    from naviertwin.core.solver_interfaces.pde_solvers import solve_heat_1d

    rng = np.random.default_rng(seed)
    params = np.zeros((n_samples, 3))
    snaps = np.zeros((n_samples, n_x))
    x = np.linspace(0, 1, n_x)

    for i in range(n_samples):
        alpha = float(10 ** rng.uniform(-2.5, -1.0))
        amps = rng.uniform(0.2, 1.0, 2)
        u0 = amps[0] * np.sin(np.pi * x) + amps[1] * np.sin(3 * np.pi * x)
        _, U = solve_heat_1d(u0, alpha=alpha, L=1.0, T=T, n_steps=n_steps)
        params[i] = [alpha, *amps]
        snaps[i] = U[-1]

    logger.info("Heat 데이터셋: %d samples, n_x=%d", n_samples, n_x)
    return {"params": params, "snapshots": snaps, "x": x}


def generate_cavity_dataset(
    n_samples: int = 10,
    nx: int = 24,
    ny: int = 24,
    tau: float = 0.8,
    seed: int | None = 0,
) -> dict[str, NDArray[np.float64]]:
    """Cavity LBM 데이터셋 — 파라미터: [u_top]."""
    from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9

    rng = np.random.default_rng(seed)
    params = np.zeros((n_samples, 1))
    snaps = np.zeros((n_samples, ny * nx))
    for i in range(n_samples):
        u_top = float(rng.uniform(0.01, 0.1))
        lbm = LBMD2Q9(nx=nx, ny=ny, tau=tau, u_top=u_top)
        s = lbm.run(n_steps=200, record_every=200)
        params[i] = u_top
        snaps[i] = s[-1, :, :, 1].ravel()
    logger.info("Cavity 데이터셋: %d samples", n_samples)
    return {"params": params, "snapshots": snaps}


__all__ = [
    "generate_burgers_dataset",
    "generate_heat_dataset",
    "generate_cavity_dataset",
]
