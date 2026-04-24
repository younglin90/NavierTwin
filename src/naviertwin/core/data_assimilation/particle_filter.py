"""Particle Filter (Bootstrap SIR).

비선형/비가우시안 동역학에 대한 시퀀셜 Monte Carlo.
    - Predict: 각 입자를 f(x) + η 로 전개
    - Weight: w_i ∝ p(y | x_i) = N(y; Hx_i, R)
    - Resample: systematic resampling (effective-N 기반 트리거)

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_assimilation.particle_filter import ParticleFilter
    >>> rng = np.random.default_rng(0)
    >>> pf = ParticleFilter(n_particles=200, state_dim=2)
    >>> pf.initialize(rng.standard_normal((200, 2)))
    >>> H = np.array([[1.0, 0.0]])
    >>> R = np.array([[0.1]])
    >>> pf.update(np.array([0.3]), H, R)
    >>> est = pf.estimate()  # posterior mean
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class ParticleFilter:
    """Bootstrap SIR filter."""

    def __init__(
        self,
        n_particles: int,
        state_dim: int,
        resample_threshold: float = 0.5,
    ) -> None:
        """Args:
            n_particles: 입자 수 N.
            state_dim: 상태 차원 n.
            resample_threshold: N_eff / N < threshold 일 때 resample.
        """
        if n_particles < 2:
            raise ValueError("n_particles >= 2 필요")
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.resample_threshold = resample_threshold

        self.particles_: NDArray[np.float64] | None = None
        self.weights_: NDArray[np.float64] | None = None

    def initialize(self, particles: NDArray[np.float64]) -> None:
        """초기 입자 배치. weights 는 uniform."""
        arr = np.asarray(particles, dtype=np.float64)
        if arr.shape != (self.n_particles, self.state_dim):
            raise ValueError(
                f"particles shape={arr.shape} != ({self.n_particles},{self.state_dim})"
            )
        self.particles_ = arr.copy()
        self.weights_ = np.full(self.n_particles, 1.0 / self.n_particles)

    def predict(
        self,
        propagator: "callable",
        process_cov: NDArray[np.float64] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """각 입자를 propagator(x) 로 전개 + process noise 추가."""
        if self.particles_ is None:
            raise RuntimeError("initialize() 를 먼저 호출하세요")
        if rng is None:
            rng = np.random.default_rng()

        new_parts = np.zeros_like(self.particles_)
        for i in range(self.n_particles):
            new_parts[i] = propagator(self.particles_[i])
        if process_cov is not None:
            noise = rng.multivariate_normal(
                mean=np.zeros(self.state_dim), cov=process_cov, size=self.n_particles
            )
            new_parts = new_parts + noise
        self.particles_ = new_parts

    def update(
        self,
        observation: NDArray[np.float64],
        H: NDArray[np.float64],
        R: NDArray[np.float64],
    ) -> None:
        """가우시안 likelihood 로 가중치 갱신 + resample."""
        if self.particles_ is None or self.weights_ is None:
            raise RuntimeError("initialize() 를 먼저 호출하세요")

        y = np.asarray(observation, dtype=np.float64).ravel()
        H = np.asarray(H, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)

        Hx = self.particles_ @ H.T  # (N, m)
        diff = y[None, :] - Hx
        try:
            R_inv = np.linalg.inv(R)
            log_det = float(np.log(max(np.linalg.det(R), 1e-300)))
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R)
            log_det = 0.0

        # log-likelihood
        log_w = -0.5 * np.einsum("ni,ij,nj->n", diff, R_inv, diff) - 0.5 * log_det
        log_w = log_w - log_w.max()  # 수치 안정화
        w = np.exp(log_w) * self.weights_
        total = float(w.sum())
        if total <= 0 or not np.isfinite(total):
            logger.warning("PF: 모든 가중치가 0 → uniform reset")
            w = np.full(self.n_particles, 1.0 / self.n_particles)
        else:
            w = w / total
        self.weights_ = w

        # Resample if N_eff low
        n_eff = 1.0 / float(np.sum(w**2))
        if n_eff < self.resample_threshold * self.n_particles:
            self._systematic_resample()

    def _systematic_resample(self) -> None:
        """Systematic resampling."""
        assert self.particles_ is not None and self.weights_ is not None
        N = self.n_particles
        positions = (np.arange(N) + np.random.random()) / N
        cumw = np.cumsum(self.weights_)
        indexes = np.searchsorted(cumw, positions)
        indexes = np.clip(indexes, 0, N - 1)
        self.particles_ = self.particles_[indexes]
        self.weights_ = np.full(N, 1.0 / N)

    def estimate(self) -> NDArray[np.float64]:
        """Weighted mean 을 반환 (posterior 추정)."""
        if self.particles_ is None or self.weights_ is None:
            raise RuntimeError("initialize() 를 먼저 호출하세요")
        return np.einsum("i,ij->j", self.weights_, self.particles_)


__all__ = ["ParticleFilter"]
