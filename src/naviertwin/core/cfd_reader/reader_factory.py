"""CFD 파일 포맷 자동 감지 팩토리 모듈.

확장자 또는 디렉토리 구조를 기반으로 적절한 :class:`BaseReader`
구현체를 자동 선택하고, 파일 읽기까지 수행하는 편의 메서드를 제공한다.

Examples:
    기본 사용법::

        from pathlib import Path
        from naviertwin.core.cfd_reader.reader_factory import ReaderFactory

        dataset = ReaderFactory.create_and_read(Path("/cases/cavity"))

    리더 수동 등록::

        @ReaderFactory.register
        class MyReader(BaseReader):
            supported_extensions = frozenset({".myext"})
            ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from naviertwin.core.cfd_reader.base import BaseReader, CFDDataset
from naviertwin.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# OpenFOAM 케이스 디렉토리 판별에 사용하는 서브디렉토리/파일 이름
_OPENFOAM_INDICATORS: frozenset[str] = frozenset(
    {"polyMesh", "constant", "system", "0"}
)


class ReaderFactory:
    """CFD 파일 포맷 자동 감지 팩토리.

    :meth:`register` 데코레이터로 등록된 리더를 확장자 맵으로 관리하며,
    :meth:`get_reader`·:meth:`create_and_read` 로 실제 리더 인스턴스를
    반환하거나 직접 데이터셋을 읽어 반환한다.

    Attributes:
        _registry: 확장자(소문자) → 리더 클래스 매핑 딕셔너리.
    """

    _registry: dict[str, type[BaseReader]] = {}

    @classmethod
    def register(cls, reader_cls: type[BaseReader]) -> type[BaseReader]:
        """리더 클래스를 확장자 레지스트리에 등록하는 데코레이터.

        ``reader_cls.supported_extensions`` 에 열거된 모든 확장자에 대해
        ``reader_cls`` 를 매핑으로 추가한다. 이미 등록된 확장자는 덮어쓴다.

        Args:
            reader_cls: 등록할 :class:`BaseReader` 구현 클래스.

        Returns:
            등록된 ``reader_cls`` 를 그대로 반환 (데코레이터 체인 지원).

        Examples:
            >>> @ReaderFactory.register
            ... class FooReader(BaseReader):
            ...     supported_extensions = frozenset({".foo"})
        """
        for ext in reader_cls.supported_extensions:
            cls._registry[ext.lower()] = reader_cls
            logger.debug(
                "리더 등록: %s → %s", ext.lower(), reader_cls.__name__
            )
        return reader_cls

    @classmethod
    def _is_openfoam_directory(cls, path: Path) -> bool:
        """디렉토리가 OpenFOAM 케이스 디렉토리인지 판별한다.

        Args:
            path: 검사할 경로.

        Returns:
            OpenFOAM 케이스 디렉토리로 판단되면 True.
        """
        if not path.is_dir():
            return False
        children = {p.name for p in path.iterdir()}
        # polyMesh 하위디렉토리가 있거나 OpenFOAM 대표 디렉토리가 2개 이상 존재
        if "polyMesh" in children:
            return True
        matched = children & _OPENFOAM_INDICATORS
        return len(matched) >= 2  # noqa: PLR2004

    @classmethod
    def get_reader(cls, path: Path) -> BaseReader:
        """경로에 적합한 리더 인스턴스를 반환한다.

        판별 우선순위:
        1. 파일 확장자가 ``.foam`` / ``.openfoam`` → OpenFOAMReader
        2. 경로가 OpenFOAM 케이스 디렉토리 구조를 가짐 → OpenFOAMReader
        3. 레지스트리에 등록된 확장자와 일치 → 해당 리더
        4. 위 모두 해당 없음 → :exc:`ValueError`

        Args:
            path: 읽을 파일 또는 디렉토리 경로.

        Returns:
            해당 경로를 처리할 수 있는 :class:`BaseReader` 인스턴스.

        Raises:
            FileNotFoundError: 경로가 존재하지 않는 경우.
            ValueError: 지원하지 않는 포맷인 경우.
        """
        if not path.exists():
            raise FileNotFoundError(f"경로가 존재하지 않습니다: {path}")

        suffix = path.suffix.lower()

        # OpenFOAM 확장자 우선 판별
        if suffix in {".foam", ".openfoam"}:
            from naviertwin.core.cfd_reader.openfoam_reader import (
                OpenFOAMReader,
            )
            logger.debug("OpenFOAMReader 선택 (확장자 기반): %s", path)
            return OpenFOAMReader()

        # 디렉토리 구조 기반 OpenFOAM 판별
        if cls._is_openfoam_directory(path):
            from naviertwin.core.cfd_reader.openfoam_reader import (
                OpenFOAMReader,
            )
            logger.debug("OpenFOAMReader 선택 (디렉토리 구조 기반): %s", path)
            return OpenFOAMReader()

        # 레지스트리 조회
        if suffix in cls._registry:
            reader_cls = cls._registry[suffix]
            logger.debug(
                "%s 선택 (레지스트리 기반): %s", reader_cls.__name__, path
            )
            return reader_cls()

        raise ValueError(
            f"지원하지 않는 파일 포맷입니다: '{suffix}' ({path})\n"
            f"지원 확장자: {sorted(cls._registry.keys())}"
        )

    @classmethod
    def create_and_read(cls, path: Path) -> CFDDataset:
        """팩토리에서 리더를 자동 선택하고 파일을 즉시 읽어 반환한다.

        :meth:`get_reader` 로 리더를 선택한 뒤 :meth:`BaseReader.read`
        를 호출하는 편의 메서드다.

        Args:
            path: 읽을 파일 또는 디렉토리 경로.

        Returns:
            파싱된 :class:`CFDDataset`.

        Raises:
            FileNotFoundError: 경로가 존재하지 않는 경우.
            ValueError: 지원하지 않는 포맷인 경우.
        """
        reader = cls.get_reader(path)
        logger.info("파일 읽기 시작: %s (리더: %s)", path, type(reader).__name__)
        dataset = reader.read(path)
        logger.info(
            "파일 읽기 완료: %d 타임스텝, 필드=%s",
            dataset.n_time_steps,
            dataset.field_names,
        )
        return dataset

    @classmethod
    def registered_extensions(cls) -> list[str]:
        """현재 레지스트리에 등록된 확장자 목록을 반환한다.

        Returns:
            정렬된 확장자 문자열 리스트.
        """
        return sorted(cls._registry.keys())
