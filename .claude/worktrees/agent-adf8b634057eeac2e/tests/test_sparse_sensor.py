"""Round 601 — sparse_sensor QR-pivot sensor placement."""

from __future__ import annotations

import numpy as np
import pytest


class TestSelectSensors:
    def test_basic_shape(self) -> None:
        from naviertwin.core.sampling.sparse_sensor import select_sensors

        rng = np.random.default_rng(0)
        U = rng.standard_normal((50, 5))
        sensors = select_sensors(U, n_sensors=8)
        assert sensors.shape == (8,)
        assert sensors.dtype in (np.intp, np.int64, np.int32)

    def test_sensors_sorted_unique(self) -> None:
        from naviertwin.core.sampling.sparse_sensor import select_sensors

        rng = np.random.default_rng(1)
        U = rng.standard_normal((100, 10))
        sensors = select_sensors(U, n_sensors=15)
        assert np.all(sensors[1:] >= sensors[:-1])
        assert len(np.unique(sensors)) == len(sensors)

    def test_sensors_in_range(self) -> None:
        from naviertwin.core.sampling.sparse_sensor import select_sensors

        rng = np.random.default_rng(2)
        U = rng.standard_normal((200, 8))
        sensors = select_sensors(U, n_sensors=20)
        assert np.all(sensors >= 0)
        assert np.all(sensors < 200)

    def test_invalid_ndim_raises(self) -> None:
        from naviertwin.core.sampling.sparse_sensor import select_sensors

        with pytest.raises(ValueError, match="2D"):
            select_sensors(np.zeros((10, 5, 3)), n_sensors=3)

    def test_invalid_n_sensors_zero(self) -> None:
        from naviertwin.core.sampling.sparse_sensor import select_sensors

        with pytest.raises(ValueError, match="n_sensors"):
            select_sensors(np.zeros((10, 5)), n_sensors=0)

    def test_invalid_n_sensors_too_large(self) -> None:
        from naviertwin.core.sampling.sparse_sensor import select_sensors

        with pytest.raises(ValueError, match="n_sensors"):
            select_sensors(np.zeros((10, 5)), n_sensors=11)

    def test_unknown_method_raises(self) -> None:
        from naviertwin.core.sampling.sparse_sensor import select_sensors

        with pytest.raises(NotImplementedError, match="method"):
            select_sensors(np.zeros((10, 5)), n_sensors=3, method="svd")

    def test_greedy_fallback(self, monkeypatch) -> None:
        import builtins

        from naviertwin.core.sampling import sparse_sensor

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "scipy.linalg":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)

        rng = np.random.default_rng(3)
        U = rng.standard_normal((30, 5))
        sensors = sparse_sensor.select_sensors(U, n_sensors=6)
        assert sensors.shape == (6,)


class TestReconstruct:
    def _make_basis(self, n_space: int = 100, n_modes: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        U, _, _ = np.linalg.svd(rng.standard_normal((n_space, n_modes)), full_matrices=False)
        return U  # orthonormal (n_space, n_modes)

    def test_reconstruct_exact_1d(self) -> None:
        from naviertwin.core.sampling.sparse_sensor import reconstruct, select_sensors

        U = self._make_basis(100, 8)
        # 진짜 신호 = basis의 선형 결합
        rng = np.random.default_rng(7)
        coeffs_true = rng.standard_normal(8)
        state_true = U @ coeffs_true

        sensors = select_sensors(U, n_sensors=10)
        y = state_true[sensors]
        state_rec = reconstruct(U, sensors, y)

        # 정확히 재구성 가능 (n_sensors >= n_modes)
        assert state_rec.shape == (100,)
        np.testing.assert_allclose(state_rec, state_true, atol=1e-8)

    def test_reconstruct_multi_sample(self) -> None:
        from naviertwin.core.sampling.sparse_sensor import reconstruct, select_sensors

        U = self._make_basis(80, 6)
        rng = np.random.default_rng(8)
        coeffs = rng.standard_normal((6, 4))
        states = U @ coeffs  # (80, 4)

        sensors = select_sensors(U, n_sensors=8)
        y = states[sensors, :]  # (8, 4)
        rec = reconstruct(U, sensors, y)
        assert rec.shape == (80, 4)
        np.testing.assert_allclose(rec, states, atol=1e-8)

    def test_reconstruct_shape_mismatch_raises(self) -> None:
        from naviertwin.core.sampling.sparse_sensor import reconstruct

        U = np.zeros((50, 5))
        sensors = np.array([0, 1, 2])
        y = np.zeros(5)  # wrong size (3 expected)
        with pytest.raises(ValueError, match="measurements"):
            reconstruct(U, sensors, y)

    def test_reconstruct_wrong_basis_ndim(self) -> None:
        from naviertwin.core.sampling.sparse_sensor import reconstruct

        with pytest.raises(ValueError, match="2D"):
            reconstruct(np.zeros((10,)), np.array([0, 1]), np.zeros(2))

    def test_reconstruct_wrong_sensors_ndim(self) -> None:
        from naviertwin.core.sampling.sparse_sensor import reconstruct

        with pytest.raises(ValueError, match="1D"):
            reconstruct(np.zeros((10, 3)), np.zeros((2, 2), dtype=int), np.zeros(2))
