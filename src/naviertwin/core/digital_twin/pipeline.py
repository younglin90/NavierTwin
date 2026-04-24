"""End-to-end NavierTwin 파이프라인 오케스트레이터.

6 단계 통합:
    1) Import — CFD 읽기 (ReaderFactory)
    2) Analyze — Q-criterion / FFT / y+
    3) Reduce — POD 또는 AE
    4) Model — Kriging 또는 FNO
    5) Twin — predict(params)
    6) Export — .ntwin / ONNX / 보고서

이 모듈은 상위 레벨 "레시피" 만 제공. 각 단계는 기존 구현 재사용.

Examples:
    >>> from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline
    >>> pipe = NavierTwinPipeline(reducer_kind="pod", n_modes=3)
    >>> # pipe.load("path/to/case.foam")
    >>> # pipe.reduce("U")
    >>> # pipe.fit_surrogate(X_params, Y_coeffs)
    >>> # result = pipe.predict_field(new_params)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineState:
    """파이프라인 상태 컨테이너."""

    dataset: Any = None
    field_name: str = ""
    snapshots: NDArray[np.float64] | None = None
    reducer: Any = None
    coeffs: NDArray[np.float64] | None = None
    surrogate: Any = None
    metrics: dict[str, float] = field(default_factory=dict)


class NavierTwinPipeline:
    """6 단계 NavierTwin 파이프라인 조율자."""

    def __init__(
        self,
        reducer_kind: str = "pod",
        n_modes: int = 5,
        surrogate_kind: str = "kriging",
    ) -> None:
        self.reducer_kind = reducer_kind
        self.n_modes = n_modes
        self.surrogate_kind = surrogate_kind
        self.state = PipelineState()

    # ------------------------------------------------------------------
    # 1) Import
    # ------------------------------------------------------------------
    def load(self, path: str) -> Any:
        from pathlib import Path

        from naviertwin.core.cfd_reader import ReaderFactory

        ds = ReaderFactory.create_and_read(Path(path))
        self.state.dataset = ds
        logger.info(
            "Pipeline 로드 완료: %d points, %d steps",
            ds.n_points, ds.n_time_steps,
        )
        return ds

    def load_snapshots(
        self, snapshots: NDArray[np.float64], field_name: str = "U"
    ) -> None:
        """데이터 리더 없이 스냅샷 배열 직접 주입 (테스트/데모 용)."""
        self.state.snapshots = np.asarray(snapshots, dtype=np.float64)
        self.state.field_name = field_name

    # ------------------------------------------------------------------
    # 3) Reduce
    # ------------------------------------------------------------------
    def reduce(self, field_name: str | None = None) -> Any:
        """POD / AE 차원축소."""
        if self.state.snapshots is None:
            if self.state.dataset is None or not field_name:
                raise RuntimeError(
                    "load() 후 field_name 제공 또는 load_snapshots() 호출 필요"
                )
            self.state.snapshots = self.state.dataset.extract_field_snapshots(
                field_name
            )
            self.state.field_name = field_name

        X = self.state.snapshots
        if self.reducer_kind == "pod":
            from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD

            red = SnapshotPOD(n_modes=self.n_modes)
        elif self.reducer_kind == "ae":
            from naviertwin.core.dimensionality_reduction.nonlinear.autoencoder import (
                Autoencoder,
            )

            red = Autoencoder(latent_dim=self.n_modes, hidden_dims=[64, 16], max_epochs=30)
        else:
            raise ValueError(f"알 수 없는 reducer_kind: {self.reducer_kind}")

        red.fit(X)
        self.state.reducer = red
        self.state.coeffs = red.encode(X)
        logger.info(
            "Reduce 완료: %s → %d 잠재 차원", self.reducer_kind, self.n_modes
        )
        return red

    # ------------------------------------------------------------------
    # 4) Model
    # ------------------------------------------------------------------
    def fit_surrogate(
        self,
        X_params: NDArray[np.float64],
        Y_coeffs: NDArray[np.float64] | None = None,
    ) -> Any:
        """파라미터 → 잠재 계수 surrogate 학습."""
        if Y_coeffs is None:
            if self.state.coeffs is None:
                raise RuntimeError("reduce() 먼저 호출 또는 Y_coeffs 전달")
            Y_coeffs = self.state.coeffs

        X = np.asarray(X_params, dtype=np.float64)
        Y = np.asarray(Y_coeffs, dtype=np.float64)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X/Y 샘플 수 불일치: {X.shape[0]} vs {Y.shape[0]}"
            )

        if self.surrogate_kind == "kriging":
            from naviertwin.core.surrogate.kriging_surrogate import KrigingSurrogate

            sur = KrigingSurrogate()
        elif self.surrogate_kind == "rbf":
            from naviertwin.core.surrogate.rbf_surrogate import RBFSurrogate

            sur = RBFSurrogate()
        else:
            raise ValueError(f"알 수 없는 surrogate_kind: {self.surrogate_kind}")

        sur.fit(X, Y)
        self.state.surrogate = sur
        logger.info("Surrogate 학습 완료: %s", self.surrogate_kind)
        return sur

    # ------------------------------------------------------------------
    # 5) Twin — predict(params) → field
    # ------------------------------------------------------------------
    def predict_field(
        self, X_params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """새 파라미터에 대한 필드 예측."""
        if self.state.surrogate is None or self.state.reducer is None:
            raise RuntimeError("fit_surrogate() 까지 완료해야 합니다")
        X = np.asarray(X_params, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        coeffs = self.state.surrogate.predict(X)
        field = self.state.reducer.decode(coeffs)
        return field

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    def validate(
        self,
        X_test: NDArray[np.float64],
        Y_true: NDArray[np.float64],
    ) -> dict[str, float]:
        from naviertwin.core.validation.metrics import compute_all_metrics

        if self.state.surrogate is None:
            raise RuntimeError("fit_surrogate() 먼저 호출하세요")
        Y_pred = self.state.surrogate.predict(X_test)
        self.state.metrics = compute_all_metrics(Y_true, Y_pred)
        return self.state.metrics

    # ------------------------------------------------------------------
    # 6) Export
    # ------------------------------------------------------------------
    def export_report(self, path: str, project: str = "NavierTwin") -> Any:
        from naviertwin.core.report.generator import ReportGenerator

        data = {
            "project": project,
            "summary": (
                f"{self.reducer_kind.upper()} → {self.surrogate_kind} 파이프라인 "
                f"(n_modes={self.n_modes})"
            ),
            "metrics": self.state.metrics,
            "model_info": {
                "reducer": self.reducer_kind,
                "n_modes": self.n_modes,
                "surrogate": self.surrogate_kind,
                "field": self.state.field_name,
            },
        }
        return ReportGenerator().render_html(data, path)


__all__ = ["NavierTwinPipeline", "PipelineState"]
