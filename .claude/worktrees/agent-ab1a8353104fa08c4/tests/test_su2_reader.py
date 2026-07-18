"""SU2Reader 테스트 모음.

SU2Reader 의 등록, meshio 폴백, ASCII 파서, .csv 사이드카 병합을 검증한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

pyvista = pytest.importorskip("pyvista", reason="pyvista 가 필요합니다")

FIXTURES = Path(__file__).parent / "fixtures"
SU2_PATH = FIXTURES / "tiny_square.su2"
CSV_PATH = FIXTURES / "tiny_square.csv"


class TestSU2ReaderRegistration:
    def test_su2_reader_registered(self) -> None:
        """SU2Reader 가 ReaderFactory 에 .su2 로 등록되어야 한다."""
        from naviertwin.core.cfd_reader import ReaderFactory
        from naviertwin.core.cfd_reader.su2_reader import SU2Reader

        assert ".su2" in ReaderFactory.registered_extensions()
        assert ReaderFactory._registry[".su2"] is SU2Reader

    def test_su2_factory_auto_detect(self) -> None:
        """ReaderFactory.get_reader(.su2) 가 SU2Reader 를 반환해야 한다."""
        from naviertwin.core.cfd_reader import ReaderFactory
        from naviertwin.core.cfd_reader.su2_reader import SU2Reader

        reader = ReaderFactory.get_reader(SU2_PATH)
        assert isinstance(reader, SU2Reader)


class TestSU2ASCIIParser:
    def test_ascii_parser_reads_points_and_cells(self) -> None:
        """네이티브 ASCII 파서가 tiny_square.su2 를 제대로 파싱해야 한다."""
        from naviertwin.core.cfd_reader.su2_reader import SU2ASCIIParser

        dataset = SU2ASCIIParser(SU2_PATH).parse()
        assert dataset.n_points == 4
        assert dataset.n_cells == 2
        assert dataset.metadata["reader"] == "SU2ASCIIParser"
        assert dataset.metadata["ndime"] == 2

    def test_reader_falls_back_to_ascii(self) -> None:
        """meshio 실패 시 ASCII 파서로 폴백해야 한다."""
        from naviertwin.core.cfd_reader.base import CFDDataset
        from naviertwin.core.cfd_reader.su2_reader import SU2Reader

        def raise_meshio(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("meshio mock 실패")

        with patch("meshio.read", side_effect=raise_meshio):
            reader = SU2Reader()
            dataset = reader.read(SU2_PATH)

        assert isinstance(dataset, CFDDataset)
        assert dataset.n_points == 4


class TestSU2MeshioPath:
    def test_meshio_happy_path(self) -> None:
        """meshio 경로가 tiny_square.su2 를 직접 파싱할 수 있어야 한다."""
        from naviertwin.core.cfd_reader.base import CFDDataset
        from naviertwin.core.cfd_reader.su2_reader import SU2Reader

        reader = SU2Reader()
        try:
            dataset = reader.read(SU2_PATH)
        except ValueError:
            pytest.skip("meshio 가 이 SU2 포맷 버전을 지원하지 않음")

        assert isinstance(dataset, CFDDataset)
        assert dataset.n_points >= 4


class TestSU2CsvSidecar:
    def test_csv_sidecar_merged(self) -> None:
        """sibling .csv 가 있으면 point_data 로 병합되어야 한다."""
        from naviertwin.core.cfd_reader.su2_reader import SU2Reader

        # meshio 는 쓰되 ASCII 파서도 가능 — 둘 중 무엇이든 .csv 병합 발생
        reader = SU2Reader()
        dataset = reader.read(SU2_PATH)

        assert "Density" in dataset.field_names
        assert "Pressure" in dataset.field_names
        # 좌표 열은 스킵
        assert "x" not in dataset.field_names
        assert dataset.metadata.get("csv_sidecar", "").endswith("tiny_square.csv")


class TestSU2ErrorHandling:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """존재하지 않는 파일은 FileNotFoundError 를 발생시켜야 한다."""
        from naviertwin.core.cfd_reader.su2_reader import SU2Reader

        reader = SU2Reader()
        with pytest.raises(FileNotFoundError):
            reader.read(tmp_path / "ghost.su2")

    def test_all_parsers_fail_raises_value_error(self, tmp_path: Path) -> None:
        """모든 파서 실패 시 ValueError 가 발생해야 한다."""
        from naviertwin.core.cfd_reader.su2_reader import SU2Reader

        bad = tmp_path / "bad.su2"
        bad.write_text("this is not su2\n")

        def raise_always(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("mock 실패")

        with patch("meshio.read", side_effect=raise_always):
            reader = SU2Reader()
            with pytest.raises(ValueError, match="SU2Reader"):
                reader.read(bad)
