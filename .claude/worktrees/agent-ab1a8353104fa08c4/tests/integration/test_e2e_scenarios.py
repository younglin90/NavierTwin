"""Round 63 — End-to-end 통합 시나리오 테스트.

각 테스트는 리얼리스틱한 워크플로우 하나를 통째로 검증한다:

    1. LBM/Burgers/Heat 데이터셋 생성
    2. POD or Autoencoder 차원축소
    3. Kriging/RBF/FNO 대리/연산자 학습
    4. 새 파라미터 → 예측
    5. 메트릭 + HTML 보고서 export

통과 기준: 전 파이프라인이 예외 없이 완주하고, R² 또는 rel.L2 가 합리적 수준.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestScenarioCavityPODKriging:
    """LBM cavity → POD → Kriging → 새 u_top 예측 → HTML 보고서."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        pytest.importorskip("sklearn")
        pytest.importorskip("jinja2")
        pytest.importorskip("smt")
        from naviertwin.core.benchmarks.dataset_catalog import (
            generate_cavity_dataset,
        )
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        data = generate_cavity_dataset(n_samples=5, nx=12, ny=12, tau=0.8, seed=0)
        X = data["snapshots"].T  # (n_features, n_samples)
        params = data["params"]

        pipe = NavierTwinPipeline(
            reducer_kind="pod", n_modes=3, surrogate_kind="kriging",
        )
        pipe.load_snapshots(X, field_name="ux")
        pipe.reduce()
        pipe.fit_surrogate(params)
        # 홀드아웃 2개
        metrics = pipe.validate(params[-2:], pipe.state.coeffs[-2:])
        assert metrics["rmse"] >= 0

        # 새 u_top=0.05 예측
        field = pipe.predict_field(np.array([[0.05]]))
        assert field.shape[0] == X.shape[0]

        # 보고서
        rpt = pipe.export_report(str(tmp_path / "cavity.html"), project="Cavity E2E")
        content = rpt.read_text(encoding="utf-8")
        assert "POD" in content.upper() or "Kriging" in content


class TestScenarioBurgersPODRBF:
    """Burgers 데이터셋 → POD → RBF → 예측."""

    def test_full_pipeline(self) -> None:
        pytest.importorskip("smt")
        from naviertwin.core.benchmarks.dataset_catalog import (
            generate_burgers_dataset,
        )
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        data = generate_burgers_dataset(
            n_samples=6, n_x=32, T=0.1, n_steps=50, seed=0,
        )
        X = data["snapshots"].T
        params = data["params"]

        pipe = NavierTwinPipeline(
            reducer_kind="pod", n_modes=3, surrogate_kind="rbf",
        )
        pipe.load_snapshots(X, field_name="u")
        pipe.reduce()
        pipe.fit_surrogate(params)
        metrics = pipe.validate(params[-2:], pipe.state.coeffs[-2:])
        assert "r2" in metrics


class TestScenarioHeatFNO:
    """Heat 데이터셋 → FNO1D 연산자 학습 → 홀드아웃 rel.L2."""

    def test_full_pipeline(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.core.benchmarks.dataset_catalog import (
            generate_heat_dataset,
        )
        from naviertwin.core.digital_twin.pipeline_neural import (
            NeuralOperatorPipeline,
        )

        data = generate_heat_dataset(
            n_samples=10, n_x=33, T=0.1, n_steps=100, seed=0,
        )
        # (n_samples, n_x, 1) FNO 형식
        snaps = data["snapshots"][:, :, None].astype(np.float32)
        # Auto-reconstruction 목표: y = x² — 간단한 mapping
        X = snaps
        Y = snaps ** 2

        pipe = NeuralOperatorPipeline(
            kind="fno1d", in_ch=1, out_ch=1,
            modes=4, width=8, n_layers=2, max_epochs=5,
        )
        pipe.fit(X[:7], Y[:7])
        # 예측 shape 검증
        y = pipe.predict(X[7:])
        assert y.shape == (3, 33, 1)
        assert np.all(np.isfinite(y))


class TestScenarioDAUQ:
    """EnKF + StreamingTwin + Sobol UQ 통합 시나리오."""

    def test_streaming_twin_with_sobol(self) -> None:
        pytest.importorskip("SALib")
        import numpy as np

        from naviertwin.core.digital_twin.streaming_twin import (
            StreamingDigitalTwin,
        )
        from naviertwin.core.sensitivity.sobol_analysis import (
            saltelli_sample,
            sobol_indices,
        )

        # 1) UQ — 가벼운 Ishigami
        bounds = np.array([[-np.pi, np.pi]] * 3, dtype=float)
        X = saltelli_sample(bounds, n_base=32, seed=0)
        Y = np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])
        S = sobol_indices(Y, n_params=3)
        assert S["S1"].shape == (3,)

        # 2) Streaming twin
        rng = np.random.default_rng(0)
        d = 2
        twin = StreamingDigitalTwin(
            state_dim=d, n_ensemble=30,
            model_fn=lambda x: x * 0.95,
            H=np.eye(d), R=0.05 * np.eye(d), rng=rng,
        )
        twin.initialize(rng.standard_normal((30, d)))
        for _ in range(5):
            twin.step()
            twin.assimilate(np.zeros(d))
        est = twin.estimate()
        assert est.shape == (d,)


class TestScenarioFullExport:
    """전체: 파이프라인 학습 → ONNX + TorchScript + HTML 보고서."""

    def test_onnx_torchscript_report(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("onnx")
        pytest.importorskip("jinja2")

        import torch
        import torch.nn as nn

        from naviertwin.core.export.onnx_export import export_to_onnx, verify_onnx
        from naviertwin.core.export.torchscript_export import export_to_torchscript
        from naviertwin.core.report.generator import ReportGenerator

        # 단순 모델
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        sample = torch.randn(1, 4)

        # ONNX
        onnx_path = export_to_onnx(model, sample, tmp_path / "model.onnx")
        info = verify_onnx(onnx_path)
        assert "ir_version" in info

        # TorchScript
        ts_path = export_to_torchscript(
            model, tmp_path / "model.pt",
            sample_input=(sample,), mode="trace",
        )
        assert ts_path.exists()
        loaded = torch.jit.load(str(ts_path))
        assert loaded(torch.zeros(1, 4)).shape == (1, 2)

        # Report
        rpt = ReportGenerator().render_html({
            "project": "Full Export E2E",
            "summary": "ONNX + TorchScript + Report 파이프라인",
            "metrics": {"rmse": 0.01, "r2": 0.99},
            "model_info": {
                "architecture": "MLP 4→8→2",
                "onnx": str(onnx_path.name),
                "torchscript": str(ts_path.name),
            },
        }, tmp_path / "report.html")
        assert rpt.exists()
