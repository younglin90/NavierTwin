"""디지털 트윈 예측 패널 — TwinEngine 을 통한 파라미터 → 필드 예측.

Signals:
    prediction_done(object): 예측 완료 시 복원된 필드 배열 발생.
    optimization_done(object): 최적화 완료 시 최적 파라미터/목적값 dict 발생.
    assimilation_done(object): 데이터 동화 quick-check 완료 시 결과 dict 발생.
    design_optimization_done(object): 설계 최적화 quick-check 결과 dict 발생.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class TwinPanel(QWidget):
    """디지털 트윈 탭 패널.

    Signals:
        prediction_done: 예측 완료 시 결과 배열과 함께 발생.
        optimization_done: 최적화 완료 시 결과 dict와 함께 발생.
    """

    prediction_done = Signal(object)
    optimization_done = Signal(object)
    assimilation_done = Signal(object)
    design_optimization_done = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._engine: Optional[object] = None
        self._external_engine_mode: bool = False
        self._n_params: int = 2
        self._param_spins: list[QDoubleSpinBox] = []
        self._setup_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI 초기화
    # ──────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 좌측 컨트롤
        left = QWidget()
        left.setFixedWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        title = QLabel("Digital Twin Prediction")
        title.setObjectName("titleLabel")
        left_layout.addWidget(title)

        # 엔진 설정
        engine_group = QGroupBox("TwinEngine 설정")
        engine_layout = QFormLayout(engine_group)

        reducer_row = QHBoxLayout()
        self._reducer_combo = QComboBox()
        self._reducer_combo.addItems(["pod", "randomized_pod"])
        reducer_row.addWidget(self._reducer_combo)
        engine_layout.addRow("Reducer:", self._reducer_combo)

        self._surrogate_combo = QComboBox()
        self._surrogate_combo.addItems(["kriging", "rbf"])
        engine_layout.addRow("Surrogate:", self._surrogate_combo)

        self._n_modes_spin = QSpinBox()
        self._n_modes_spin.setRange(1, 200)
        self._n_modes_spin.setValue(10)
        engine_layout.addRow("POD 모드 수:", self._n_modes_spin)

        left_layout.addWidget(engine_group)

        # 파라미터 입력
        param_group = QGroupBox("파라미터 입력")
        param_outer_layout = QVBoxLayout(param_group)

        n_params_row = QHBoxLayout()
        n_params_row.addWidget(QLabel("파라미터 수:"))
        self._n_params_spin = QSpinBox()
        self._n_params_spin.setRange(1, 20)
        self._n_params_spin.setValue(2)
        self._n_params_spin.valueChanged.connect(self._rebuild_param_inputs)
        n_params_row.addWidget(self._n_params_spin)
        param_outer_layout.addLayout(n_params_row)

        # 스크롤 가능한 파라미터 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        self._param_widget = QWidget()
        self._param_layout = QFormLayout(self._param_widget)
        scroll.setWidget(self._param_widget)
        param_outer_layout.addWidget(scroll)

        left_layout.addWidget(param_group)

        # 엔진 로드/저장
        io_group = QGroupBox("엔진 저장/로드")
        io_layout = QHBoxLayout(io_group)

        self._save_btn = QPushButton("저장")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_engine)
        io_layout.addWidget(self._save_btn)

        self._load_btn = QPushButton("로드")
        self._load_btn.clicked.connect(self._load_engine)
        io_layout.addWidget(self._load_btn)

        left_layout.addWidget(io_group)

        # 예측 버튼
        self._predict_btn = QPushButton("예측 실행")
        self._predict_btn.setObjectName("primaryButton")
        self._predict_btn.clicked.connect(self._run_predict)
        left_layout.addWidget(self._predict_btn)

        # 최적화
        opt_group = QGroupBox("Optimization")
        opt_form = QFormLayout(opt_group)

        self._optimizer_combo = QComboBox()
        self._optimizer_combo.addItems(["SurrogateOptimizer", "BayesianOptimizer"])
        opt_form.addRow("Optimizer:", self._optimizer_combo)

        self._objective_combo = QComboBox()
        self._objective_combo.addItems([
            "min field mean",
            "min field norm",
            "match target scalar",
        ])
        opt_form.addRow("Objective:", self._objective_combo)

        self._target_spin = QDoubleSpinBox()
        self._target_spin.setRange(-1e9, 1e9)
        self._target_spin.setDecimals(6)
        self._target_spin.setValue(0.0)
        opt_form.addRow("Target:", self._target_spin)

        self._bound_low_spin = QDoubleSpinBox()
        self._bound_low_spin.setRange(-1e9, 1e9)
        self._bound_low_spin.setDecimals(4)
        self._bound_low_spin.setValue(0.0)
        opt_form.addRow("Lower bound:", self._bound_low_spin)

        self._bound_high_spin = QDoubleSpinBox()
        self._bound_high_spin.setRange(-1e9, 1e9)
        self._bound_high_spin.setDecimals(4)
        self._bound_high_spin.setValue(1.0)
        opt_form.addRow("Upper bound:", self._bound_high_spin)

        self._n_initial_spin = QSpinBox()
        self._n_initial_spin.setRange(2, 200)
        self._n_initial_spin.setValue(6)
        opt_form.addRow("Initial evals:", self._n_initial_spin)

        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(0, 200)
        self._max_iter_spin.setValue(8)
        opt_form.addRow("Max iter:", self._max_iter_spin)

        self._optimize_btn = QPushButton("최적화 실행")
        self._optimize_btn.clicked.connect(self._run_optimize)
        opt_form.addRow(self._optimize_btn)

        left_layout.addWidget(opt_group)

        assim_group = QGroupBox("Assimilation Quick Check")
        assim_form = QFormLayout(assim_group)

        self._assim_method_combo = QComboBox()
        self._assim_method_combo.addItems(["4D-Var", "Particle Filter", "UKF"])
        assim_form.addRow("Method:", self._assim_method_combo)

        self._assim_state_dim_spin = QSpinBox()
        self._assim_state_dim_spin.setRange(1, 10)
        self._assim_state_dim_spin.setValue(2)
        assim_form.addRow("State dim:", self._assim_state_dim_spin)

        self._assim_steps_spin = QSpinBox()
        self._assim_steps_spin.setRange(1, 50)
        self._assim_steps_spin.setValue(5)
        assim_form.addRow("Steps:", self._assim_steps_spin)

        self._assim_particles_spin = QSpinBox()
        self._assim_particles_spin.setRange(20, 2000)
        self._assim_particles_spin.setValue(200)
        assim_form.addRow("Particles:", self._assim_particles_spin)

        self._assim_noise_spin = QDoubleSpinBox()
        self._assim_noise_spin.setRange(1e-6, 10.0)
        self._assim_noise_spin.setDecimals(6)
        self._assim_noise_spin.setValue(0.05)
        assim_form.addRow("Obs noise:", self._assim_noise_spin)

        self._assim_btn = QPushButton("동화 quick-check")
        self._assim_btn.clicked.connect(self._run_assimilation)
        assim_form.addRow(self._assim_btn)

        left_layout.addWidget(assim_group)

        design_group = QGroupBox("Design Optimization Quick Check")
        design_form = QFormLayout(design_group)

        self._design_method_combo = QComboBox()
        self._design_method_combo.addItems(["NSGA-II Pareto", "SIMP Topology"])
        design_form.addRow("Method:", self._design_method_combo)

        self._design_size_spin = QSpinBox()
        self._design_size_spin.setRange(2, 50)
        self._design_size_spin.setValue(8)
        design_form.addRow("Dim/Grid nx:", self._design_size_spin)

        self._design_iter_spin = QSpinBox()
        self._design_iter_spin.setRange(1, 100)
        self._design_iter_spin.setValue(8)
        design_form.addRow("Generations/iters:", self._design_iter_spin)

        self._design_volume_spin = QDoubleSpinBox()
        self._design_volume_spin.setRange(0.1, 0.9)
        self._design_volume_spin.setDecimals(3)
        self._design_volume_spin.setValue(0.5)
        design_form.addRow("Volume fraction:", self._design_volume_spin)

        self._design_btn = QPushButton("설계 최적화 quick-check")
        self._design_btn.clicked.connect(self._run_design_optimization)
        design_form.addRow(self._design_btn)

        left_layout.addWidget(design_group)

        # 데모 학습 버튼
        self._demo_btn = QPushButton("데모 학습 & 예측")
        self._demo_btn.clicked.connect(self._run_demo)
        left_layout.addWidget(self._demo_btn)

        left_layout.addStretch()
        layout.addWidget(left)

        # 우측: 결과
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        result_group = QGroupBox("예측 결과")
        result_layout = QVBoxLayout(result_group)
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        result_layout.addWidget(self._result_text)
        right_layout.addWidget(result_group)

        self._status_label = QLabel("TwinEngine을 로드하거나 데모를 실행하세요.")
        self._status_label.setObjectName("subtitleLabel")
        right_layout.addWidget(self._status_label)

        layout.addWidget(right, stretch=1)

        # 초기 파라미터 입력 빌드
        self._rebuild_param_inputs(2)

    def _rebuild_param_inputs(self, n: int) -> None:
        """파라미터 수에 맞게 입력 스핀박스를 재생성한다."""
        # 기존 위젯 제거
        while self._param_layout.rowCount() > 0:
            self._param_layout.removeRow(0)
        self._param_spins.clear()

        self._n_params = n
        for i in range(n):
            spin = QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setValue(0.5)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            self._param_layout.addRow(f"param_{i}:", spin)
            self._param_spins.append(spin)

    # ──────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────

    def set_engine(self, engine: object) -> None:
        """외부에서 TwinEngine을 설정한다."""
        self._engine = engine
        self._set_external_engine_mode(True)
        self._save_btn.setEnabled(True)
        self._status_label.setText("TwinEngine 준비 완료.")
        self._sync_engine_settings(engine)
        # 학습된 surrogate 입력 차원에 맞춰 파라미터 입력 UI를 자동 동기화한다.
        try:
            surrogate = getattr(engine, "surrogate", None)
            n_params = int(getattr(surrogate, "input_dim", 0))
            if n_params > 0 and n_params != self._n_params_spin.value():
                self._n_params_spin.setValue(n_params)
        except Exception:
            pass
        self._log(f"TwinEngine 설정: {type(engine).__name__}")

    def _set_external_engine_mode(self, enabled: bool) -> None:
        """외부에서 주입된 엔진 모드 여부를 설정한다."""
        self._external_engine_mode = enabled
        self._reducer_combo.setEnabled(not enabled)
        self._surrogate_combo.setEnabled(not enabled)
        self._n_modes_spin.setEnabled(not enabled)
        self._demo_btn.setEnabled(not enabled)

    def _sync_engine_settings(self, engine: object) -> None:
        """외부 엔진의 설정값을 UI에 동기화한다."""
        reducer_type = str(getattr(engine, "reducer_type", "")).lower()
        surrogate_type = str(getattr(engine, "surrogate_type", "")).lower()
        n_modes = int(getattr(engine, "n_modes", self._n_modes_spin.value()))

        if reducer_type:
            idx = self._reducer_combo.findText(reducer_type)
            if idx < 0:
                self._reducer_combo.addItem(reducer_type)
                idx = self._reducer_combo.findText(reducer_type)
            if idx >= 0:
                self._reducer_combo.setCurrentIndex(idx)

        if surrogate_type:
            idx = self._surrogate_combo.findText(surrogate_type)
            if idx < 0:
                self._surrogate_combo.addItem(surrogate_type)
                idx = self._surrogate_combo.findText(surrogate_type)
            if idx >= 0:
                self._surrogate_combo.setCurrentIndex(idx)

        if n_modes > 0:
            self._n_modes_spin.setValue(n_modes)

    # ──────────────────────────────────────────────────────────────────
    # 슬롯
    # ──────────────────────────────────────────────────────────────────

    def _run_predict(self) -> None:
        if self._engine is None:
            self._log("[WARN] TwinEngine이 없습니다. 먼저 로드하거나 데모를 실행하세요.")
            return
        try:
            params = np.array([[spin.value() for spin in self._param_spins]], dtype=float)
            result = self._engine.predict(params)  # type: ignore[union-attr]
            self._log(f"예측 완료: shape={result.shape}, min={result.min():.4g}, max={result.max():.4g}")
            self._status_label.setText("예측 완료.")
            self.prediction_done.emit(result)
        except Exception as exc:
            self._log(f"[ERROR] {exc}")

    def _run_optimize(self) -> None:
        if self._engine is None:
            self._log("[WARN] TwinEngine이 없습니다. 먼저 로드하거나 데모를 실행하세요.")
            return

        low = float(self._bound_low_spin.value())
        high = float(self._bound_high_spin.value())
        if not low < high:
            self._log("[WARN] 최적화 bounds는 lower < upper 이어야 합니다.")
            return

        n_dims = max(1, int(self._n_params_spin.value()))
        bounds = np.tile(np.array([[low, high]], dtype=np.float64), (n_dims, 1))
        objective_name = self._objective_combo.currentText()

        try:
            optimizer = self._build_optimizer(bounds)
            objective = self._build_objective(objective_name)
            x_best, f_best = optimizer.minimize(objective)  # type: ignore[attr-defined]
            x_best = np.asarray(x_best, dtype=float).reshape(-1)
            for spin, value in zip(self._param_spins, x_best):
                spin.setValue(float(value))
            n_eval = len(getattr(optimizer, "y_", []))
            result = {
                "optimizer": self._optimizer_combo.currentText(),
                "objective": objective_name,
                "x_best": x_best,
                "f_best": float(f_best),
                "n_eval": int(n_eval),
            }
            self._log(
                "최적화 완료: "
                f"objective={objective_name}, f_best={float(f_best):.6g}, "
                f"x_best={np.array2string(x_best, precision=4)}, n_eval={n_eval}"
            )
            self._status_label.setText("최적화 완료.")
            self.optimization_done.emit(result)
        except Exception as exc:
            self._log(f"[ERROR] 최적화 실패: {exc}")

    def _run_assimilation(self) -> None:
        method = self._assim_method_combo.currentText()
        n_state = int(self._assim_state_dim_spin.value())
        n_steps = int(self._assim_steps_spin.value())
        obs_noise = float(self._assim_noise_spin.value())

        try:
            if method == "4D-Var":
                result = self._run_four_dvar_demo(n_state, n_steps, obs_noise)
            elif method == "Particle Filter":
                result = self._run_particle_filter_demo(n_state, n_steps, obs_noise)
            else:
                result = self._run_ukf_demo(n_state, n_steps, obs_noise)

            estimate = np.asarray(result["estimate"], dtype=float)
            self._log(
                "동화 quick-check 완료: "
                f"method={method}, error={float(result['error']):.6g}, "
                f"estimate={np.array2string(estimate, precision=4)}"
            )
            self._status_label.setText(f"{method} 동화 완료.")
            self.assimilation_done.emit(result)
        except Exception as exc:
            self._log(f"[ERROR] 동화 실패: {exc}")

    def _run_design_optimization(self) -> None:
        method = self._design_method_combo.currentText()
        try:
            if method == "NSGA-II Pareto":
                result = self._run_nsga2_demo()
            else:
                result = self._run_simp_demo()
            self._log(self._format_design_result(result))
            self._status_label.setText(f"{method} 완료.")
            self.design_optimization_done.emit(result)
        except Exception as exc:
            self._log(f"[ERROR] 설계 최적화 실패: {exc}")

    def _run_nsga2_demo(self) -> dict[str, object]:
        from naviertwin.core.optimization.moo_optimizer import NSGA2

        n_dims = min(int(self._design_size_spin.value()), 10)
        n_gen = int(self._design_iter_spin.value())
        pop_size = max(12, 4 * n_dims)
        bounds = np.tile(np.array([[-1.0, 1.0]], dtype=float), (n_dims, 1))

        def objective(x: np.ndarray) -> list[float]:
            return [
                float(np.sum((x - 0.2) ** 2)),
                float(np.sum((x + 0.5) ** 2)),
            ]

        optimizer = NSGA2(
            bounds=bounds,
            n_obj=2,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=0,
        )
        pareto, objectives = optimizer.optimize(objective)
        return {
            "method": "NSGA-II Pareto",
            "pareto": pareto,
            "objectives": objectives,
            "pareto_count": int(pareto.shape[0]),
            "n_dims": int(n_dims),
            "n_gen": int(n_gen),
        }

    def _run_simp_demo(self) -> dict[str, object]:
        from naviertwin.core.optimization.topology_opt import simp_2d

        nx = int(self._design_size_spin.value())
        ny = max(2, nx // 2)
        n_iter = int(self._design_iter_spin.value())
        vol_frac = float(self._design_volume_spin.value())
        density = simp_2d(nx=nx, ny=ny, vol_frac=vol_frac, n_iter=n_iter)
        return {
            "method": "SIMP Topology",
            "density": density,
            "shape": tuple(int(v) for v in density.shape),
            "volume_fraction": float(np.mean(density)),
            "density_min": float(np.min(density)),
            "density_max": float(np.max(density)),
            "n_iter": int(n_iter),
        }

    @staticmethod
    def _format_design_result(result: dict[str, object]) -> str:
        method = str(result.get("method", "Design Optimization"))
        if method == "NSGA-II Pareto":
            objectives = np.asarray(result["objectives"], dtype=float)
            best = objectives[np.argmin(objectives[:, 0])]
            return (
                "설계 최적화 완료: "
                f"method={method}, pareto={int(result['pareto_count'])}, "
                f"best_f=[{best[0]:.4g}, {best[1]:.4g}]"
            )
        density = np.asarray(result["density"], dtype=float)
        return (
            "설계 최적화 완료: "
            f"method={method}, shape={density.shape}, "
            f"vol={float(result['volume_fraction']):.4g}, "
            f"range=[{float(result['density_min']):.4g}, "
            f"{float(result['density_max']):.4g}]"
        )

    def _run_four_dvar_demo(
        self, n_state: int, n_steps: int, obs_noise: float
    ) -> dict[str, object]:
        from naviertwin.core.data_assimilation.four_dvar import four_dvar_linear

        x_true, M, H, Y = self._assimilation_scenario(n_state, n_steps, obs_noise)
        x_b = np.zeros(n_state, dtype=float)
        B = np.eye(n_state, dtype=float)
        R = (obs_noise**2 + 1e-9) * np.eye(n_state, dtype=float)
        estimate = four_dvar_linear(x_b, B, Y, H, R, M)
        return self._assimilation_result("4D-Var", estimate, x_true, n_steps)

    def _run_particle_filter_demo(
        self, n_state: int, n_steps: int, obs_noise: float
    ) -> dict[str, object]:
        from naviertwin.core.data_assimilation.particle_filter import ParticleFilter

        x_true, M, H, Y = self._assimilation_scenario(n_state, n_steps, obs_noise)
        n_particles = int(self._assim_particles_spin.value())
        rng = np.random.default_rng(0)
        pf = ParticleFilter(n_particles=n_particles, state_dim=n_state)
        particles = rng.normal(loc=x_true, scale=0.25, size=(n_particles, n_state))
        pf.initialize(particles)
        Q = (obs_noise**2 + 1e-9) * np.eye(n_state, dtype=float)
        R = (obs_noise**2 + 1e-9) * np.eye(n_state, dtype=float)
        for obs in Y:
            pf.predict(lambda x: M @ x, process_cov=Q, rng=rng)
            pf.update(obs, H, R)
        estimate = pf.estimate()
        truth = np.linalg.matrix_power(M, n_steps) @ x_true
        result = self._assimilation_result("Particle Filter", estimate, truth, n_steps)
        result["n_particles"] = n_particles
        return result

    def _run_ukf_demo(
        self, n_state: int, n_steps: int, obs_noise: float
    ) -> dict[str, object]:
        from naviertwin.core.data_assimilation.ukf import ukf_step

        x_true, M, H, Y = self._assimilation_scenario(n_state, n_steps, obs_noise)
        x = np.zeros(n_state, dtype=float)
        P = np.eye(n_state, dtype=float)
        Q = (obs_noise**2 + 1e-9) * np.eye(n_state, dtype=float)
        R = (obs_noise**2 + 1e-9) * np.eye(n_state, dtype=float)
        for obs in Y:
            x, P = ukf_step(
                x,
                P,
                lambda state: M @ state,
                lambda state: H @ state,
                z=obs,
                Q=Q,
                R=R,
            )
        truth = np.linalg.matrix_power(M, n_steps) @ x_true
        result = self._assimilation_result("UKF", x, truth, n_steps)
        result["cov_trace"] = float(np.trace(P))
        return result

    @staticmethod
    def _assimilation_scenario(
        n_state: int, n_steps: int, obs_noise: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_true = np.linspace(1.0, 0.4, n_state, dtype=float)
        M = 0.92 * np.eye(n_state, dtype=float)
        if n_state > 1:
            M += 0.03 * np.diag(np.ones(n_state - 1), k=1)
        H = np.eye(n_state, dtype=float)
        x = x_true.copy()
        observations: list[np.ndarray] = []
        deterministic_noise = obs_noise * np.linspace(-0.5, 0.5, n_state)
        for _ in range(n_steps):
            x = M @ x
            observations.append(H @ x + deterministic_noise)
        return x_true, M, H, np.vstack(observations)

    @staticmethod
    def _assimilation_result(
        method: str, estimate: np.ndarray, truth: np.ndarray, n_steps: int
    ) -> dict[str, object]:
        estimate = np.asarray(estimate, dtype=float).reshape(-1)
        truth = np.asarray(truth, dtype=float).reshape(-1)
        return {
            "method": method,
            "estimate": estimate,
            "truth": truth,
            "error": float(np.linalg.norm(estimate - truth)),
            "n_state": int(truth.size),
            "n_steps": int(n_steps),
        }

    def _build_optimizer(self, bounds: np.ndarray) -> object:
        optimizer_name = self._optimizer_combo.currentText()
        n_initial = int(self._n_initial_spin.value())
        max_iter = int(self._max_iter_spin.value())
        if optimizer_name == "BayesianOptimizer":
            from naviertwin.core.optimization.bayesian_opt import BayesianOptimizer

            return BayesianOptimizer(
                bounds=bounds,
                n_initial=n_initial,
                max_iter=max_iter,
                seed=0,
            )

        from naviertwin.core.optimization.surrogate_opt import SurrogateOptimizer

        return SurrogateOptimizer(
            bounds=bounds,
            surrogate_kind="rbf",
            n_initial=n_initial,
            max_iter=max_iter,
            seed=0,
        )

    def _build_objective(self, objective_name: str) -> Callable[[np.ndarray], float]:
        target = float(self._target_spin.value())

        def objective(x: np.ndarray) -> float:
            assert self._engine is not None
            params = np.asarray(x, dtype=float).reshape(1, -1)
            field = np.asarray(self._engine.predict(params), dtype=float)  # type: ignore[union-attr]
            if objective_name == "min field norm":
                return float(np.linalg.norm(field))
            mean = float(np.mean(field))
            if objective_name == "match target scalar":
                return float((mean - target) ** 2)
            return mean

        return objective

    def _run_demo(self) -> None:
        """데모: 랜덤 데이터로 TwinEngine을 학습하고 예측한다."""
        if self._external_engine_mode:
            self._log("[WARN] 외부 엔진 모드에서는 데모 학습을 실행할 수 없습니다.")
            return
        try:
            from naviertwin.core.digital_twin.twin_engine import TwinEngine

            n_params = self._n_params_spin.value()
            n_modes = self._n_modes_spin.value()
            reducer = self._reducer_combo.currentText()
            surrogate = self._surrogate_combo.currentText()

            # 데모 데이터
            rng = np.random.default_rng(0)
            n_features = 200
            n_samples = 30
            snapshots = rng.random((n_features, n_samples))
            params = rng.random((n_samples, n_params))

            engine = TwinEngine(
                reducer_type=reducer,
                surrogate_type=surrogate,
                n_modes=min(n_modes, n_samples - 1),
            )
            engine.fit(snapshots, params)
            self._engine = engine
            self._save_btn.setEnabled(True)

            # 예측
            test_params = np.array([[spin.value() for spin in self._param_spins]], dtype=float)
            if test_params.shape[1] != n_params:
                test_params = rng.random((1, n_params))

            result = engine.predict(test_params)
            self._log(
                f"[Demo] TwinEngine 학습+예측 완료\n"
                f"  reducer={reducer}, surrogate={surrogate}, n_modes={n_modes}\n"
                f"  output shape={result.shape}, range=[{result.min():.4g}, {result.max():.4g}]\n"
            )
            self._status_label.setText("데모 완료.")
            self.prediction_done.emit(result)

        except Exception as exc:
            self._log(f"[ERROR] {exc}")

    def _save_engine(self) -> None:
        if self._engine is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "TwinEngine 저장", "twin_engine.pkl", "Pickle (*.pkl)"
        )
        if path:
            try:
                self._engine.save(Path(path))  # type: ignore[union-attr]
                self._log(f"엔진 저장: {path}")
            except Exception as exc:
                self._log(f"[ERROR] 저장 실패: {exc}")

    def _load_engine(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "TwinEngine 로드", "", "Pickle (*.pkl)"
        )
        if path:
            try:
                from naviertwin.core.digital_twin.twin_engine import TwinEngine
                engine = TwinEngine.load(Path(path))
                self.set_engine(engine)
                self._log(f"엔진 로드: {path}")
            except Exception as exc:
                self._log(f"[ERROR] 로드 실패: {exc}")

    def _log(self, msg: str) -> None:
        self._result_text.append(msg)
        sb = self._result_text.verticalScrollBar()
        sb.setValue(sb.maximum())
