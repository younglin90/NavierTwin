"""Langevin dynamics + Euler-Maruyama SDE 적분.

Langevin sampler:
    x_{t+1} = x_t + (ε/2) · ∇log p(x_t) + sqrt(ε) · ξ_t,   ξ_t ~ N(0, I)

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.generative.langevin import langevin_sample
    >>> def score(x):
    ...     return -x  # p(x) ∝ exp(-0.5 x²) → N(0, 1)
    >>> rng = np.random.default_rng(0)
    >>> samples = langevin_sample(score, x0=np.array([2.0]),
    ...                           n_steps=2000, step_size=0.01, rng=rng)
    >>> abs(float(samples[-500:].mean())) < 0.5
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def langevin_sample(
    score_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    n_steps: int = 1000,
    step_size: float = 0.01,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Score-based Langevin 샘플러.

    Args:
        score_fn: x → ∇_x log p(x). shape 유지.
        x0: 초기 상태.
        n_steps: 반복 수.
        step_size: ε.
        rng: numpy Generator.

    Returns:
        sample trajectory (n_steps, *x0.shape).
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x0, dtype=np.float64).copy()
    out = np.zeros((n_steps, *x.shape))
    sqrt_eps = np.sqrt(step_size)
    t = 0
    while t < n_steps:
        noise = rng.standard_normal(x.shape)
        g = np.asarray(score_fn(x), dtype=np.float64)
        x = x + 0.5 * step_size * g + sqrt_eps * noise
        out[t] = x
        t += 1
    logger.info("Langevin: %d steps, ε=%.4g", n_steps, step_size)
    return out


def euler_maruyama(
    drift: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
    diffusion: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    t_span: tuple[float, float],
    n_steps: int,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Euler-Maruyama SDE 적분기.

        dx = b(t, x) dt + σ(t, x) dW_t

    Returns:
        (times, trajectory shape=(n_steps+1, *x0.shape)).
    """
    if rng is None:
        rng = np.random.default_rng()
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    x = np.asarray(x0, dtype=np.float64).copy()
    traj = np.zeros((n_steps + 1, *x.shape))
    traj[0] = x
    times = np.linspace(t0, tf, n_steps + 1)
    sqrt_dt = np.sqrt(dt)
    k = 0
    while k < n_steps:
        t = times[k]
        dW = rng.standard_normal(x.shape) * sqrt_dt
        b = np.asarray(drift(t, x), dtype=np.float64)
        s = np.asarray(diffusion(t, x), dtype=np.float64)
        x = x + b * dt + s * dW
        traj[k + 1] = x
        k += 1
    return times, traj


__all__ = ["langevin_sample", "euler_maruyama"]
