"""Round 31 — EchoStateNetwork + PDE-FIND."""

from __future__ import annotations

import numpy as np
import pytest


class TestESN:
    def test_sine_prediction(self) -> None:
        from naviertwin.core.time_series.esn.esn import EchoStateNetwork

        t = np.linspace(0, 20, 500)
        seq = np.column_stack([np.sin(t), np.cos(t)])
        esn = EchoStateNetwork(
            n_features=2, reservoir_size=80,
            spectral_radius=0.95, sparsity=0.15, seed=0,
        )
        esn.fit(seq[:350], seq[1:351], warmup=50)
        y = esn.predict(seq[350:-1])
        assert y.shape == (149, 2)
        # 예측이 유한
        assert np.all(np.isfinite(y))

    def test_shape_mismatch(self) -> None:
        from naviertwin.core.time_series.esn.esn import EchoStateNetwork

        esn = EchoStateNetwork(n_features=3, reservoir_size=10)
        with pytest.raises(ValueError):
            esn.fit(np.zeros((5, 3)), np.zeros((6, 3)))


class TestPDEFind:
    def test_heat_equation_recovered(self) -> None:
        from naviertwin.core.flow_analysis.modal.pde_find import pde_find_1d
        from naviertwin.core.solver_interfaces.pde_solvers import solve_heat_1d

        x = np.linspace(0, 1, 33)
        u0 = np.sin(np.pi * x) + 0.3 * np.sin(3 * np.pi * x)
        t, U = solve_heat_1d(u0, alpha=0.05, L=1.0, T=0.3, n_steps=500)
        # 내부만 사용 (boundary 효과 제거)
        res = pde_find_1d(U[:, 1:-1], t, x[1:-1], threshold=0.02)
        # 발견된 방정식에 U_xx 가 포함돼야
        assert "U_xx" in res["equation"]

    def test_wrong_shape(self) -> None:
        from naviertwin.core.flow_analysis.modal.pde_find import pde_find_1d

        with pytest.raises(ValueError):
            pde_find_1d(np.zeros(10), np.zeros(10), np.zeros(10))
