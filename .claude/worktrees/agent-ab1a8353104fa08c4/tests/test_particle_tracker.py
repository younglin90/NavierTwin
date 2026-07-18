"""Round 231 — Lagrangian tracker."""

from __future__ import annotations

import numpy as np


class TestTracker:
    def test_uniform_flow(self) -> None:
        from naviertwin.core.analysis.particle_tracker import track_particles_2d

        u = np.ones((10, 10))
        v = np.zeros((10, 10))
        seeds = np.array([[0.2, 0.5], [0.3, 0.5]])
        trails = track_particles_2d(u, v, seeds, Lx=1.0, Ly=1.0,
                                    dt=0.01, n_steps=20)
        assert trails.shape == (2, 21, 2)
        # x direction moved
        assert trails[0, -1, 0] > trails[0, 0, 0]
        # y unchanged
        assert abs(trails[0, -1, 1] - 0.5) < 1e-6

    def test_residence_time(self) -> None:
        from naviertwin.core.analysis.particle_tracker import (
            residence_time,
            track_particles_2d,
        )

        u = np.ones((10, 10))
        v = np.zeros((10, 10))
        seeds = np.array([[0.0, 0.5]])
        trails = track_particles_2d(u, v, seeds, Lx=1.0, Ly=1.0,
                                    dt=0.01, n_steps=50)
        rt = residence_time(trails, box=(0.0, 0.0, 0.5, 1.0), dt=0.01)
        assert rt[0] > 0
