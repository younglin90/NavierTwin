"""Round 422 — particle smoother."""

from __future__ import annotations

import numpy as np


class TestPS:
    def test_uniform_transition_keeps_uniform(self) -> None:
        from naviertwin.core.data_assimilation.particle_smoother import (
            smooth_particles,
        )

        N, M = 4, 10
        particles = np.zeros((N, M))
        weights = np.full((N, M), 1.0 / M)
        # uniform transition
        ws = smooth_particles(particles, weights, lambda a, b: 1.0)
        # weights stay uniform
        assert np.allclose(ws, 1.0 / M, atol=1e-10)
