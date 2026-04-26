"""Round 602 — SensorDMDPipeline fit/reconstruct coverage."""

from __future__ import annotations

import numpy as np
import pytest


class TestSensorDMDPipeline:
    def _make_data(self, n_space: int = 80, n_snap: int = 30, seed: int = 0):
        rng = np.random.default_rng(seed)
        # low-rank signal + noise
        U = rng.standard_normal((n_space, 6))
        V = rng.standard_normal((6, n_snap))
        return U @ V + 0.01 * rng.standard_normal((n_space, n_snap))

    def test_fit_sets_basis_and_sensors(self) -> None:
        from naviertwin.core.sampling.sensor_dmd_pipeline import SensorDMDPipeline

        X = self._make_data()
        pipe = SensorDMDPipeline(n_modes=5, n_sensors=8)
        pipe.fit(X)
        assert pipe.is_fitted
        assert pipe.basis is not None
        assert pipe.basis.shape[0] == 80
        assert pipe.basis.shape[1] <= 5
        assert pipe.sensors is not None

    def test_reconstruct_shape_2d(self) -> None:
        from naviertwin.core.sampling.sensor_dmd_pipeline import SensorDMDPipeline

        X = self._make_data()
        pipe = SensorDMDPipeline(n_modes=6, n_sensors=10)
        pipe.fit(X)
        rec = pipe.reconstruct_from_sensors(X[:, :3])
        assert rec.shape == (80, 3)

    def test_reconstruct_shape_1d(self) -> None:
        from naviertwin.core.sampling.sensor_dmd_pipeline import SensorDMDPipeline

        X = self._make_data()
        pipe = SensorDMDPipeline(n_modes=6, n_sensors=10)
        pipe.fit(X)
        rec = pipe.reconstruct_from_sensors(X[:, 0])
        assert rec.shape == (80,)

    def test_reconstruct_before_fit_raises(self) -> None:
        from naviertwin.core.sampling.sensor_dmd_pipeline import SensorDMDPipeline

        pipe = SensorDMDPipeline(n_modes=5)
        with pytest.raises(RuntimeError, match="fit"):
            pipe.reconstruct_from_sensors(np.zeros(80))

    def test_fit_1d_raises(self) -> None:
        from naviertwin.core.sampling.sensor_dmd_pipeline import SensorDMDPipeline

        pipe = SensorDMDPipeline(n_modes=5)
        with pytest.raises(ValueError, match="2D"):
            pipe.fit(np.zeros(80))

    def test_invalid_n_modes_raises(self) -> None:
        from naviertwin.core.sampling.sensor_dmd_pipeline import SensorDMDPipeline

        with pytest.raises(ValueError, match="n_modes"):
            SensorDMDPipeline(n_modes=0)

    def test_energy_fraction(self) -> None:
        from naviertwin.core.sampling.sensor_dmd_pipeline import SensorDMDPipeline

        X = self._make_data()
        pipe = SensorDMDPipeline(n_modes=5, n_sensors=8)
        pipe.fit(X)
        ef = pipe.energy_fraction()
        assert ef.shape[0] <= 5
        assert abs(ef.sum() - 1.0) < 1e-10

    def test_energy_fraction_before_fit_raises(self) -> None:
        from naviertwin.core.sampling.sensor_dmd_pipeline import SensorDMDPipeline

        pipe = SensorDMDPipeline(n_modes=5)
        with pytest.raises(RuntimeError, match="fit"):
            pipe.energy_fraction()

    def test_default_n_sensors(self) -> None:
        from naviertwin.core.sampling.sensor_dmd_pipeline import SensorDMDPipeline

        pipe = SensorDMDPipeline(n_modes=7)
        assert pipe.n_sensors == 9  # n_modes + 2
