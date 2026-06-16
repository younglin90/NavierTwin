"""CFD 리더 추상 기반 클래스 및 데이터 컨테이너 정의.

모든 구체적인 CFD 파일 포맷 리더는 :class:`BaseReader`를 상속하고
:meth:`BaseReader.read` 메서드를 구현해야 한다.

Examples:
    커스텀 리더 구현::

        from pathlib import Path
        from naviertwin.core.cfd_reader.base import BaseReader, CFDDataset

        class MyFormatReader(BaseReader):
            def read(self, path: Path) -> CFDDataset:
                # 파일 파싱 로직
                mesh = ...
                return CFDDataset(
                    mesh=mesh,
                    time_steps=[0.0, 0.1, 0.2],
                    field_names=["U", "p"],
                    metadata={"solver": "MyFoam"},
                )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class CFDDataset:
    """CFD 시뮬레이션 데이터 컨테이너.

    내부적으로 PyVista UnstructuredGrid를 메쉬 표현으로 사용한다.
    멀티 타임스텝 데이터를 포함할 수 있으며, 각 타임스텝의 필드 데이터는
    ``mesh.point_data`` 또는 ``mesh.cell_data`` 딕셔너리에 저장된다.

    Attributes:
        mesh: PyVista UnstructuredGrid 메쉬. 점 및 셀 데이터를 포함한다.
        time_steps: 타임스텝 목록 (단위: 초). 정렬되어 있어야 한다.
        field_names: 메쉬에 존재하는 물리량 이름 목록.
            예: ["U", "p", "T", "k", "epsilon"].
        metadata: 파일 포맷, 솔버 정보 등 임의의 부가 정보.
    """

    mesh: Any  # pv.UnstructuredGrid — TYPE_CHECKING 블록 밖에서는 Any 사용
    time_steps: list[float] = field(default_factory=list)
    field_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """초기화 후 유효성 검사.

        Raises:
            TypeError: mesh가 None인 경우.
        """
        if self.mesh is None:
            raise TypeError("mesh는 None일 수 없습니다.")

    @property
    def n_time_steps(self) -> int:
        """타임스텝 수를 반환한다."""
        return len(self.time_steps)

    @property
    def n_points(self) -> int:
        """메쉬의 점 개수를 반환한다."""
        return int(self.mesh.n_points)

    @property
    def n_cells(self) -> int:
        """메쉬의 셀 개수를 반환한다."""
        return int(self.mesh.n_cells)

    def extract_field_snapshots(self, field_name: str) -> Any:
        """필드명을 기준으로 (n_features, n_steps) 스냅샷 행렬을 추출한다.

        다중 타임스텝 저장 형식이 명확하지 않은 경우 보수적으로 판단해
        단일 스냅샷 (n_features, 1)로 폴백한다.

        Args:
            field_name: 추출할 필드 이름.

        Returns:
            shape = (n_features, n_steps) 의 ndarray.

        Raises:
            ValueError: 필드가 메쉬에 없는 경우.
        """
        import numpy as np

        mesh = self.mesh
        time_series_fields = self.metadata.get("time_series_fields")
        if (
            isinstance(time_series_fields, dict)
            and field_name in time_series_fields
        ):
            arr = np.asarray(time_series_fields[field_name], dtype=float)
            locations = self.metadata.get("time_series_locations", {})
            location = (
                locations.get(field_name, "point")
                if isinstance(locations, dict)
                else "point"
            )
            expected_per_step = self.n_cells if location == "cell" else self.n_points
        elif field_name in mesh.point_data:
            arr = np.asarray(mesh.point_data[field_name], dtype=float)
            expected_per_step = self.n_points
        elif field_name in mesh.cell_data:
            arr = np.asarray(mesh.cell_data[field_name], dtype=float)
            expected_per_step = self.n_cells
        else:
            raise ValueError(f"필드 '{field_name}'가 메쉬에 없습니다.")

        n_steps = max(1, self.n_time_steps)
        return self._reshape_snapshots(arr, expected_per_step, n_steps)

    @staticmethod
    def _reshape_snapshots(arr: Any, expected_per_step: int, n_steps: int) -> Any:
        """필드 배열을 스냅샷 행렬로 변환한다."""
        import numpy as np

        arr = np.asarray(arr, dtype=float)
        if n_steps <= 1:
            return CFDDataset._to_single_snapshot(arr)

        if arr.ndim == 1:
            if expected_per_step > 0 and arr.size == expected_per_step * n_steps:
                return arr.reshape(n_steps, expected_per_step).T
            return arr.reshape(-1, 1)

        if arr.ndim == 2:
            if (
                arr.shape[0] == n_steps
                and expected_per_step > 0
                and arr.shape[1] == expected_per_step
            ):
                return arr.T
            if (
                arr.shape[0] == expected_per_step
                and arr.shape[1] == n_steps
            ):
                return arr
            if expected_per_step > 0 and arr.shape[0] == expected_per_step * n_steps:
                reshaped = arr.reshape(n_steps, expected_per_step, arr.shape[1])
                return np.linalg.norm(reshaped, axis=-1).T
            return CFDDataset._to_single_snapshot(arr)

        if arr.ndim == 3:
            if (
                arr.shape[0] == n_steps
                and expected_per_step > 0
                and arr.shape[1] == expected_per_step
            ):
                return np.linalg.norm(arr, axis=-1).T
            return CFDDataset._to_single_snapshot(arr)

        return CFDDataset._to_single_snapshot(arr)

    @staticmethod
    def _to_single_snapshot(arr: Any) -> Any:
        """배열을 단일 스냅샷 (n_features, 1)로 변환한다."""
        import numpy as np

        arr = np.asarray(arr, dtype=float)
        if arr.ndim > 1:
            arr = np.linalg.norm(arr, axis=-1)
        return arr.reshape(-1, 1)


class BaseReader(ABC):
    """CFD 파일 포맷 리더의 추상 기반 클래스.

    모든 CFD 리더는 이 클래스를 상속하여 :meth:`read` 메서드를 구현해야 한다.
    팩토리 패턴과 함께 사용되며, ``reader_factory.py``에서 확장자에 따라
    적절한 리더를 자동 선택한다.

    Attributes:
        supported_extensions: 이 리더가 지원하는 파일 확장자 집합.
            예: {".foam", ".OpenFOAM"}.
    """

    supported_extensions: frozenset[str] = frozenset()

    @abstractmethod
    def read(self, path: Path) -> CFDDataset:
        """CFD 파일(또는 디렉토리)을 읽어 :class:`CFDDataset`으로 반환한다.

        Args:
            path: 읽을 파일 또는 디렉토리 경로.

        Returns:
            파싱된 CFD 데이터셋.

        Raises:
            FileNotFoundError: 경로가 존재하지 않는 경우.
            ValueError: 파일 포맷이 올바르지 않은 경우.
            NotImplementedError: 구체 클래스에서 구현되지 않은 경우.
        """
        ...

    def can_read(self, path: Path) -> bool:
        """해당 경로의 파일을 이 리더가 읽을 수 있는지 확인한다.

        기본 구현은 파일 확장자를 :attr:`supported_extensions`와 비교한다.
        필요에 따라 하위 클래스에서 재정의할 수 있다.

        Args:
            path: 확인할 파일 경로.

        Returns:
            읽을 수 있으면 True, 아니면 False.
        """
        return path.suffix.lower() in self.supported_extensions

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(extensions={set(self.supported_extensions)})"
        )
