"""GUI 위젯 모듈.

    - VtkViewer: PyVista/pyvistaqt 기반 CFD 3D viewer
    - ModelCompareWidget: 모델 메트릭 비교 대시보드
    - LossCurveWidget: 학습 loss curve 시각화
"""

from naviertwin.gui.widgets.loss_curve_widget import LossCurveWidget
from naviertwin.gui.widgets.model_compare_widget import ModelCompareWidget
from naviertwin.gui.widgets.vtk_viewer import VtkViewer

__all__ = ["VtkViewer", "ModelCompareWidget", "LossCurveWidget"]
