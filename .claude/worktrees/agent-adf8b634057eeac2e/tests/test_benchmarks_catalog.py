"""Round 45 — 벤치마크 데이터셋 레지스트리."""

from __future__ import annotations

import numpy as np


class TestBenchmarks:
    def test_burgers_dataset(self) -> None:
        from naviertwin.core.benchmarks.dataset_catalog import (
            generate_burgers_dataset,
        )

        data = generate_burgers_dataset(n_samples=5, n_x=16, T=0.1, n_steps=50, seed=0)
        assert data["params"].shape == (5, 4)
        assert data["snapshots"].shape == (5, 16)
        assert np.all(np.isfinite(data["snapshots"]))

    def test_heat_dataset(self) -> None:
        from naviertwin.core.benchmarks.dataset_catalog import (
            generate_heat_dataset,
        )

        data = generate_heat_dataset(n_samples=4, n_x=17, T=0.1, n_steps=100, seed=0)
        assert data["params"].shape == (4, 3)
        assert data["snapshots"].shape == (4, 17)
        # 경계 0
        assert np.allclose(data["snapshots"][:, 0], 0, atol=1e-10)
        assert np.allclose(data["snapshots"][:, -1], 0, atol=1e-10)

    def test_cavity_dataset(self) -> None:
        from naviertwin.core.benchmarks.dataset_catalog import (
            generate_cavity_dataset,
        )

        data = generate_cavity_dataset(n_samples=3, nx=8, ny=8, tau=0.8, seed=0)
        assert data["params"].shape == (3, 1)
        assert data["snapshots"].shape == (3, 64)
