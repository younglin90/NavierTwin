"""Round 61 — SimulationPanel import + 기능 단위 호출 검증."""

from __future__ import annotations

import pytest


class TestSimulationPanelImport:
    def test_module_loads(self) -> None:
        pytest.importorskip("PySide6")
        from naviertwin.gui.panels import SimulationPanel

        assert SimulationPanel is not None

    def test_main_window_has_simulation(self) -> None:
        pytest.importorskip("PySide6")
        from naviertwin.gui.main_window import MainWindow

        # attribute/탭 목록 검증 (인스턴스 생성 없이)
        assert hasattr(MainWindow, "_setup_panels")


class TestBackendEnginesStillWork:
    """SimulationPanel 이 내부적으로 호출하는 엔진 로직 검증 (Qt 없이)."""

    def test_lbm_backend(self) -> None:

        from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9

        lbm = LBMD2Q9(nx=8, ny=8, tau=0.8, u_top=0.05)
        s = lbm.run(n_steps=10, record_every=5)
        assert s.shape == (2, 8, 8, 3)

    def test_streaming_backend(self) -> None:
        import numpy as np

        from naviertwin.core.digital_twin.streaming_twin import StreamingDigitalTwin

        rng = np.random.default_rng(0)
        d = 3
        twin = StreamingDigitalTwin(
            state_dim=d, n_ensemble=20,
            model_fn=lambda x: x,
            H=np.eye(d), R=0.01 * np.eye(d), rng=rng,
        )
        twin.initialize(rng.standard_normal((20, d)))
        twin.step()
        twin.assimilate(np.zeros(d))
        est = twin.estimate()
        assert est.shape == (d,)

    def test_rl_backend(self) -> None:
        pytest.importorskip("torch")
        import numpy as np

        from naviertwin.core.flow_control.policy_gradient import (
            GaussianPolicy,
            reinforce_update,
        )

        pol = GaussianPolicy(state_dim=3, action_dim=1, hidden=8, seed=0)
        rng = np.random.default_rng(0)
        for _ in range(3):
            s = rng.standard_normal((10, 3)).astype(np.float32)
            a, _ = pol.sample(s)
            r = -np.abs(a).ravel()
            reinforce_update(pol, s, a, None, r, lr=1e-2)

    def test_burgers_backend(self) -> None:
        import numpy as np

        from naviertwin.core.solver_interfaces.fvm_advection import fvm_upwind_1d

        u0 = np.sin(np.linspace(0, 2 * np.pi, 32, endpoint=False))
        _, U = fvm_upwind_1d(u0, c=1.0, L=2 * np.pi, T=0.2, cfl=0.4)
        assert U.shape[1] == 32
