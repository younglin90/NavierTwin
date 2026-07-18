"""실시간 시뮬레이션 패널 — LBM / Streaming twin / RL control 런처.

Signals:
    simulation_done(str, object): 시뮬 완료 시 (종류, 결과 dict) 발생.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class SimulationPanel(QWidget):
    """실시간 시뮬레이션 런처 — LBM cavity / Streaming digital twin / RL flow control."""

    simulation_done = Signal(str, object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 좌측 제어 패널
        left = QWidget()
        left.setFixedWidth(280)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(8)

        title = QLabel("Simulation")
        title.setObjectName("titleLabel")
        lv.addWidget(title)

        # 종류 선택
        kind_group = QGroupBox("엔진")
        kind_layout = QVBoxLayout(kind_group)
        self._kind_combo = QComboBox()
        self._kind_combo.addItems([
            "LBM D2Q9 cavity",
            "Streaming Digital Twin",
            "RL flow control (REINFORCE)",
            "Burgers FVM (upwind)",
        ])
        self._kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        kind_layout.addWidget(self._kind_combo)
        lv.addWidget(kind_group)

        # 파라미터 스택
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_lbm_params())
        self._stack.addWidget(self._build_streaming_params())
        self._stack.addWidget(self._build_rl_params())
        self._stack.addWidget(self._build_burgers_params())
        lv.addWidget(self._stack)

        self._run_btn = QPushButton("시뮬레이션 실행")
        self._run_btn.setObjectName("primaryButton")
        self._run_btn.clicked.connect(self._run)
        lv.addWidget(self._run_btn)
        lv.addStretch()

        layout.addWidget(left)

        # 우측 로그/결과
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        log_group = QGroupBox("결과 / 로그")
        log_layout = QVBoxLayout(log_group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        log_layout.addWidget(self._log)
        rv.addWidget(log_group)
        layout.addWidget(right, stretch=1)

    # ------------------------------------------------------------------
    # 파라미터 페이지들
    # ------------------------------------------------------------------

    def _build_lbm_params(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(4, 4, 4, 4)
        self._lbm_nx = QSpinBox()
        self._lbm_nx.setRange(8, 256)
        self._lbm_nx.setValue(32)
        f.addRow("nx:", self._lbm_nx)
        self._lbm_ny = QSpinBox()
        self._lbm_ny.setRange(8, 256)
        self._lbm_ny.setValue(32)
        f.addRow("ny:", self._lbm_ny)
        self._lbm_tau = QDoubleSpinBox()
        self._lbm_tau.setRange(0.51, 2.0)
        self._lbm_tau.setSingleStep(0.01)
        self._lbm_tau.setValue(0.8)
        f.addRow("τ:", self._lbm_tau)
        self._lbm_u_top = QDoubleSpinBox()
        self._lbm_u_top.setRange(0.001, 0.2)
        self._lbm_u_top.setSingleStep(0.005)
        self._lbm_u_top.setValue(0.05)
        self._lbm_u_top.setDecimals(4)
        f.addRow("u_top:", self._lbm_u_top)
        self._lbm_steps = QSpinBox()
        self._lbm_steps.setRange(10, 5000)
        self._lbm_steps.setValue(200)
        f.addRow("n_steps:", self._lbm_steps)
        return w

    def _build_streaming_params(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self._st_dim = QSpinBox()
        self._st_dim.setRange(1, 100)
        self._st_dim.setValue(3)
        f.addRow("state_dim:", self._st_dim)
        self._st_ens = QSpinBox()
        self._st_ens.setRange(5, 500)
        self._st_ens.setValue(40)
        f.addRow("ensemble:", self._st_ens)
        self._st_steps = QSpinBox()
        self._st_steps.setRange(5, 1000)
        self._st_steps.setValue(20)
        f.addRow("n_steps:", self._st_steps)
        return w

    def _build_rl_params(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self._rl_state = QSpinBox()
        self._rl_state.setRange(1, 32)
        self._rl_state.setValue(4)
        f.addRow("state_dim:", self._rl_state)
        self._rl_action = QSpinBox()
        self._rl_action.setRange(1, 16)
        self._rl_action.setValue(1)
        f.addRow("action_dim:", self._rl_action)
        self._rl_episodes = QSpinBox()
        self._rl_episodes.setRange(1, 200)
        self._rl_episodes.setValue(20)
        f.addRow("episodes:", self._rl_episodes)
        return w

    def _build_burgers_params(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self._bg_N = QSpinBox()
        self._bg_N.setRange(16, 512)
        self._bg_N.setValue(64)
        f.addRow("N:", self._bg_N)
        self._bg_c = QDoubleSpinBox()
        self._bg_c.setRange(-5.0, 5.0)
        self._bg_c.setValue(1.0)
        f.addRow("c (wave speed):", self._bg_c)
        self._bg_T = QDoubleSpinBox()
        self._bg_T.setRange(0.01, 10.0)
        self._bg_T.setValue(1.0)
        f.addRow("T:", self._bg_T)
        return w

    # ------------------------------------------------------------------
    # 슬롯
    # ------------------------------------------------------------------

    def _on_kind_changed(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)

    def _run(self) -> None:
        idx = self._kind_combo.currentIndex()
        try:
            if idx == 0:
                result = self._run_lbm()
                self.simulation_done.emit("lbm_cavity", result)
            elif idx == 1:
                result = self._run_streaming()
                self.simulation_done.emit("streaming_twin", result)
            elif idx == 2:
                result = self._run_rl()
                self.simulation_done.emit("rl", result)
            elif idx == 3:
                result = self._run_burgers()
                self.simulation_done.emit("burgers", result)
            else:
                self._log.append("[WARN] 알 수 없는 엔진")
                return
            self._log.append(f"[완료] 엔진 {self._kind_combo.currentText()} — {result.get('summary', '')}")
        except Exception as e:  # noqa: BLE001
            self._log.append(f"[ERROR] {e}")

    def _run_lbm(self) -> dict[str, Any]:
        from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9

        lbm = LBMD2Q9(
            nx=self._lbm_nx.value(), ny=self._lbm_ny.value(),
            tau=self._lbm_tau.value(), u_top=self._lbm_u_top.value(),
        )
        snaps = lbm.run(
            n_steps=self._lbm_steps.value(),
            record_every=max(1, self._lbm_steps.value() // 4),
        )
        last = snaps[-1]
        return {
            "snapshots": snaps,
            "summary": (
                f"snapshots={snaps.shape}, ux_max={float(last[..., 1].max()):.4g}, "
                f"rho_mean={float(last[..., 0].mean()):.4g}"
            ),
        }

    def _run_streaming(self) -> dict[str, Any]:

        from naviertwin.core.digital_twin.streaming_twin import StreamingDigitalTwin

        rng = np.random.default_rng(0)
        d = self._st_dim.value()
        N = self._st_ens.value()
        n_steps = self._st_steps.value()
        # 합성 안정 선형 모델
        A = np.eye(d) * 0.95
        H = np.eye(d)
        R = 0.01 * np.eye(d)
        twin = StreamingDigitalTwin(
            state_dim=d, n_ensemble=N,
            model_fn=lambda x: A @ x, H=H, R=R, rng=rng,
        )
        truth = rng.standard_normal(d)
        twin.initialize(truth + rng.standard_normal((N, d)))
        errs = []
        step = 0
        while step < n_steps:
            twin.step()
            y = truth + 0.05 * rng.standard_normal(d)
            twin.assimilate(y)
            errs.append(float(np.linalg.norm(twin.estimate() - truth)))
            step += 1
        return {
            "estimate": twin.estimate(),
            "uncertainty": twin.uncertainty(),
            "errors": errs,
            "summary": f"steps={n_steps}, 최종 err={errs[-1]:.4g}",
        }

    def _run_rl(self) -> dict[str, Any]:

        from naviertwin.core.flow_control.policy_gradient import (
            GaussianPolicy,
            reinforce_update,
        )

        rng = np.random.default_rng(0)
        sd = self._rl_state.value()
        ad = self._rl_action.value()
        episodes = self._rl_episodes.value()
        pol = GaussianPolicy(state_dim=sd, action_dim=ad, hidden=16, seed=0)
        losses = []
        episode = 0
        while episode < episodes:
            s = rng.standard_normal((20, sd)).astype(np.float32)
            a, _ = pol.sample(s)
            r = -np.abs(a).ravel()  # 작은 액션 선호
            losses.append(reinforce_update(pol, s, a, None, r, lr=1e-2))
            episode += 1
        return {
            "losses": losses,
            "summary": f"episodes={episodes}, 최종 loss={losses[-1]:.4g}",
        }

    def _run_burgers(self) -> dict[str, Any]:

        from naviertwin.core.solver_interfaces.fvm_advection import fvm_upwind_1d

        N = self._bg_N.value()
        c = self._bg_c.value()
        T = self._bg_T.value()
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        u0 = np.sin(x)
        t, U = fvm_upwind_1d(u0, c=c, L=2 * np.pi, T=T, cfl=0.4)
        return {
            "times": t,
            "U": U,
            "summary": f"N={N}, c={c}, 최종 max|u|={float(np.abs(U[-1]).max()):.4g}",
        }


__all__ = ["SimulationPanel"]
