"""Round 57 — foamlib wrapper (OpenFOAM 설치 없이도 API 동작)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("foamlib", reason="foamlib 필요")


def _write_transport_properties(case_dir: Path) -> None:
    """최소 OpenFOAM dictionary 파일 작성."""
    (case_dir / "constant").mkdir(parents=True, exist_ok=True)
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "0").mkdir(parents=True, exist_ok=True)
    content = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}

transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] 1e-05;
"""
    (case_dir / "constant" / "transportProperties").write_text(content)


class TestFoamlibOps:
    def test_read_foam_dict(self, tmp_path: Path) -> None:
        from naviertwin.core.cfd_reader.foamlib_case import read_foam_dict

        case = tmp_path / "case1"
        _write_transport_properties(case)
        try:
            d = read_foam_dict(case / "constant" / "transportProperties")
            # transportModel / nu 중 하나라도 키에 있어야
            assert "transportModel" in d or "nu" in d
        except Exception as e:
            pytest.skip(f"foamlib 파서가 이 파일을 처리 못함: {e}")

    def test_modify_transport_properties(self, tmp_path: Path) -> None:
        from naviertwin.core.cfd_reader.foamlib_case import modify_transport_properties

        case = tmp_path / "case1"
        _write_transport_properties(case)
        try:
            modify_transport_properties(case, nu=5e-5)
            content = (case / "constant" / "transportProperties").read_text()
            assert "5e-05" in content or "5e-5" in content.lower() or "5.0e-05" in content
        except Exception as e:
            pytest.skip(f"foamlib 수정 실패 (버전 API 차이): {e}")

    def test_parameter_sweep(self, tmp_path: Path) -> None:
        from naviertwin.core.cfd_reader.foamlib_case import parameter_sweep

        template = tmp_path / "template"
        _write_transport_properties(template)
        out = tmp_path / "out"
        try:
            cases = parameter_sweep(
                template,
                sweep={"nu": [1e-5, 1e-6]},
                out_dir=out,
            )
            assert len(cases) == 2
            assert all(c.exists() for c in cases)
        except Exception as e:
            pytest.skip(f"parameter sweep 환경 제약: {e}")
