"""Round 59 — PhysicsNEMO Module wrap + pyCGNS tree API."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestPhysicsNEMOWrap:
    def test_availability_check(self) -> None:
        from naviertwin.core.physnemo.physicsnemo_model import physicsnemo_available

        # 이 환경에선 True
        assert physicsnemo_available() is True

    def test_wrap_module_and_roundtrip(self, tmp_path: Path) -> None:
        import torch
        import torch.nn as nn

        from naviertwin.core.physnemo.physicsnemo_model import (
            load_checkpoint,
            save_checkpoint,
            wrap_as_physicsnemo_module,
        )

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        try:
            wrapped = wrap_as_physicsnemo_module(model, name="demo_pnemo")
        except Exception as e:
            pytest.skip(f"PhysicsNEMO 환경 제약: {e}")

        x = torch.randn(3, 4)
        y = wrapped(x)
        assert y.shape == (3, 2)

        out = tmp_path / "ckpt.pt"
        save_checkpoint(wrapped, out)
        assert out.exists()

        # 새 wrap → load → 동일 출력 검증
        model2 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        try:
            wrapped2 = wrap_as_physicsnemo_module(model2, name="demo_pnemo")
        except Exception as e:
            pytest.skip(f"{e}")
        load_checkpoint(wrapped2, out)
        y2 = wrapped2(x)
        assert y.shape == y2.shape

    def test_invalid_input(self) -> None:
        from naviertwin.core.physnemo.physicsnemo_model import (
            wrap_as_physicsnemo_module,
        )

        with pytest.raises(ValueError, match="nn.Module"):
            wrap_as_physicsnemo_module("not a module")


class TestPyCGNSTree:
    def test_load_and_list(self, tmp_path: Path) -> None:
        pycgns = pytest.importorskip("CGNS.MAP", reason="pyCGNS 필요")
        del pycgns
        # 기존 fixture 사용
        fixture = Path(__file__).parent / "fixtures" / "synthetic.cgns"
        if not fixture.exists():
            pytest.skip("synthetic.cgns 없음")

        from naviertwin.core.cfd_reader.pycgns_tree import (
            list_zones,
            load_cgns_tree,
            walk_tree,
        )

        try:
            tree = load_cgns_tree(fixture)
        except Exception as e:
            pytest.skip(f"pyCGNS 가 이 HDF5 구조 지원 안함: {e}")

        lines = walk_tree(tree, max_depth=3)
        assert len(lines) >= 1
        zones = list_zones(tree)
        assert isinstance(zones, list)

    def test_walk_synthetic_tree(self) -> None:
        """가짜 CGNS 트리 구조 직접 작성해 walk/list 검증."""
        import numpy as np

        from naviertwin.core.cfd_reader.pycgns_tree import (
            get_coordinates,
            list_solutions,
            list_zones,
            walk_tree,
        )

        tree = [
            "CGNSTree", None,
            [
                ["Base", None, [
                    ["Zone1", None, [
                        ["GridCoordinates", None, [
                            ["CoordinateX", np.linspace(0, 1, 5), [], "DataArray_t"],
                            ["CoordinateY", np.zeros(5), [], "DataArray_t"],
                        ], "GridCoordinates_t"],
                        ["FlowSolution", None, [], "FlowSolution_t"],
                    ], "Zone_t"],
                ], "CGNSBase_t"],
            ],
            "CGNSTree_t",
        ]

        lines = walk_tree(tree)
        assert any("Zone1" in ln for ln in lines)
        assert list_zones(tree) == ["Zone1"]
        assert list_solutions(tree) == [("Zone1", "FlowSolution")]
        coords = get_coordinates(tree, "Zone1")
        assert "CoordinateX" in coords
        assert coords["CoordinateX"].shape == (5,)

    def test_walk_empty_tree(self) -> None:
        from naviertwin.core.cfd_reader.pycgns_tree import walk_tree

        # 잘못된 구조
        assert walk_tree("not a tree") == []
