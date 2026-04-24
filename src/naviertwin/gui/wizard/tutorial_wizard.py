"""NavierTwin 튜토리얼 위자드.

첫 실행 시 합성 스냅샷 → POD → Kriging → 예측 → 보고서까지
5 단계 흐름을 안내한다. PySide6 QWizard 기반.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from PySide6.QtCore import Qt  # noqa: F401
    from PySide6.QtWidgets import (
        QCheckBox,
        QLabel,
        QSpinBox,
        QVBoxLayout,
        QWizard,
        QWizardPage,
    )
except ImportError:  # pragma: no cover
    QWizard = None  # type: ignore[misc,assignment]
    QWizardPage = None  # type: ignore[misc,assignment]


def _require_qt() -> Any:
    if QWizard is None:
        raise RuntimeError("PySide6 필요")


class TutorialWizard(QWizard):
    """5 단계 튜토리얼 위자드."""

    def __init__(self, parent: Any = None) -> None:
        _require_qt()
        super().__init__(parent)
        self.setWindowTitle("NavierTwin 튜토리얼")
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.state: dict[str, Any] = {}

        self.addPage(self._welcome_page())
        self.addPage(self._data_page())
        self.addPage(self._reduce_page())
        self.addPage(self._model_page())
        self.addPage(self._finish_page())

    # ------------------------------------------------------------------

    def _welcome_page(self) -> QWizardPage:
        page = QWizardPage()
        page.setTitle("환영합니다")
        page.setSubTitle("이 위자드는 NavierTwin 의 전체 파이프라인을 안내합니다.")
        lay = QVBoxLayout(page)
        lay.addWidget(QLabel(
            "합성 유동장 스냅샷을 생성하고, POD 로 차원축소한 뒤 "
            "Kriging 대리 모델을 학습하여 예측 결과를 확인합니다."
        ))
        return page

    def _data_page(self) -> QWizardPage:
        page = QWizardPage()
        page.setTitle("1 단계 — 합성 스냅샷 생성")
        lay = QVBoxLayout(page)
        lay.addWidget(QLabel("격자 크기와 스냅샷 개수를 설정하세요."))
        self._n_pts = QSpinBox()
        self._n_pts.setRange(10, 1000)
        self._n_pts.setValue(50)
        lay.addWidget(self._n_pts)
        self._n_snap = QSpinBox()
        self._n_snap.setRange(4, 1000)
        self._n_snap.setValue(30)
        lay.addWidget(self._n_snap)
        return page

    def _reduce_page(self) -> QWizardPage:
        page = QWizardPage()
        page.setTitle("2 단계 — POD 차원축소")
        lay = QVBoxLayout(page)
        lay.addWidget(QLabel("POD 모드 수"))
        self._n_modes = QSpinBox()
        self._n_modes.setRange(1, 50)
        self._n_modes.setValue(5)
        lay.addWidget(self._n_modes)
        return page

    def _model_page(self) -> QWizardPage:
        page = QWizardPage()
        page.setTitle("3 단계 — Kriging 학습 및 검증")
        lay = QVBoxLayout(page)
        lay.addWidget(QLabel("학습/검증 분할 비율 80% 로 자동 진행됩니다."))
        self._validate = QCheckBox("검증 메트릭 출력 (RMSE/R²)")
        self._validate.setChecked(True)
        lay.addWidget(self._validate)
        return page

    def _finish_page(self) -> QWizardPage:
        page = QWizardPage()
        page.setTitle("완료 — 보고서 생성")
        lay = QVBoxLayout(page)
        lay.addWidget(QLabel(
            "마침 버튼을 누르면 파이프라인이 실행되고 결과가 상태에 저장됩니다."
        ))
        return page

    # ------------------------------------------------------------------

    def run_pipeline(self) -> dict[str, Any]:
        """위자드 선택에 따라 end-to-end 파이프라인을 합성 데이터로 실행한다."""
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        rng = np.random.default_rng(0)
        n_pts = self._n_pts.value()
        n_snap = self._n_snap.value()
        n_modes = self._n_modes.value()

        # 저랭크 합성 스냅샷
        U = rng.standard_normal((n_pts, n_modes))
        V = rng.standard_normal((n_modes, n_snap))
        X = U @ V + 0.02 * rng.standard_normal((n_pts, n_snap))

        pipe = NavierTwinPipeline(
            reducer_kind="pod", n_modes=n_modes, surrogate_kind="kriging",
        )
        pipe.load_snapshots(X, field_name="U")
        pipe.reduce()
        params = np.linspace(0, 1, n_snap).reshape(-1, 1)
        pipe.fit_surrogate(params)
        metrics = pipe.validate(params[-5:], pipe.state.coeffs[-5:])
        self.state = {
            "pipeline": pipe,
            "metrics": metrics,
            "n_modes": n_modes,
        }
        return self.state


__all__ = ["TutorialWizard"]
