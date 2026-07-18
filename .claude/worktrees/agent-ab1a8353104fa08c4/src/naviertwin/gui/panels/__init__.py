"""GUI 패널 모듈.

    - ImportPanel: CFD 파일 가져오기 패널
    - AnalyzePanel: 유동 분석 패널
    - ReducePanel: 차원 축소 패널
    - ModelPanel: 모델 학습/평가 패널
    - TwinPanel: 디지털 트윈 예측 패널
    - ExplainabilityPanel: 모델 설명 패널
    - ExportPanel: 결과 내보내기 패널
    - SimulationPanel: 실시간 시뮬레이션 패널
    - LibraryPanel: 라이브러리/기능 통합 브라우저
    - PostProcessPanel: 후처리 도구 패널
"""

from naviertwin.gui.panels.analyze_panel import AnalyzePanel
from naviertwin.gui.panels.explainability_panel import ExplainabilityPanel
from naviertwin.gui.panels.export_panel import ExportPanel
from naviertwin.gui.panels.import_panel import ImportPanel
from naviertwin.gui.panels.library_panel import LibraryPanel
from naviertwin.gui.panels.model_panel import ModelPanel
from naviertwin.gui.panels.postproc_panel import PostProcessPanel
from naviertwin.gui.panels.reduce_panel import ReducePanel
from naviertwin.gui.panels.simulation_panel import SimulationPanel
from naviertwin.gui.panels.twin_panel import TwinPanel

__all__ = [
    "ImportPanel",
    "AnalyzePanel",
    "ReducePanel",
    "ModelPanel",
    "TwinPanel",
    "ExplainabilityPanel",
    "ExportPanel",
    "SimulationPanel",
    "LibraryPanel",
    "PostProcessPanel",
]
