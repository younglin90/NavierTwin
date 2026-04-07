"""유동 분석기 추상 기반 클래스.

모든 유동 분석 알고리즘(:mod:`vortex`, :mod:`modal`, :mod:`statistics` 등)은
:class:`BaseFlowAnalyzer`를 상속하고 :meth:`compute` 메서드를 구현한다.

Examples:
    커스텀 분석기 구현::

        from naviertwin.core.flow_analysis.base import BaseFlowAnalyzer

        class QCriterionAnalyzer(BaseFlowAnalyzer):
            def compute(self, mesh, **kwargs):
                threshold = kwargs.get("threshold", 0.0)
                # Q-criterion 계산 로직
                mesh["Q"] = ...
                return mesh
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyvista as pv


class BaseFlowAnalyzer(ABC):
    """유동 분석기의 추상 기반 클래스.

    메쉬 데이터를 입력으로 받아 계산된 결과를 메쉬 데이터로 반환하는
    ``compute`` 인터페이스를 강제한다. 결과는 원본 메쉬에 새 필드를
    추가하거나 새 메쉬를 반환하는 방식으로 제공된다.

    Attributes:
        name: 분석기의 고유 이름. 로그 및 GUI 표시에 사용된다.
        description: 분석기에 대한 간략한 설명.
    """

    name: str = "BaseFlowAnalyzer"
    description: str = ""

    @abstractmethod
    def compute(self, mesh: Any, **kwargs: Any) -> Any:
        """메쉬 데이터에 유동 분석을 수행하고 결과를 반환한다.

        Args:
            mesh: 분석할 PyVista UnstructuredGrid 메쉬.
                point_data 또는 cell_data에 속도장, 압력장 등이 포함되어 있어야 한다.
            **kwargs: 분석기별 추가 인수.
                예: threshold=0.5, field_name="U", n_modes=10.

        Returns:
            결과 필드가 추가된 PyVista UnstructuredGrid.
            원본 메쉬를 수정하거나 새 메쉬를 반환할 수 있다.

        Raises:
            KeyError: 필요한 필드가 메쉬에 없는 경우.
            ValueError: kwargs의 인수가 유효하지 않은 경우.
        """
        ...

    def validate_mesh(self, mesh: Any, required_fields: list[str]) -> None:
        """메쉬에 필요한 필드가 모두 존재하는지 검사한다.

        Args:
            mesh: 검사할 PyVista UnstructuredGrid.
            required_fields: 필수 필드 이름 목록.

        Raises:
            KeyError: 필요한 필드가 메쉬에 없는 경우.
        """
        available = set(mesh.point_data.keys()) | set(mesh.cell_data.keys())
        missing = [f for f in required_fields if f not in available]
        if missing:
            raise KeyError(
                f"메쉬에 다음 필드가 없습니다: {missing}. "
                f"사용 가능한 필드: {sorted(available)}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
