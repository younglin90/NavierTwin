"""Round 439 — PVD extended."""

from __future__ import annotations


class TestPVDExt:
    def test_grouped(self, tmp_path) -> None:
        from naviertwin.core.visualization.pvd_extended import write_pvd_grouped

        p = tmp_path / "x.pvd"
        write_pvd_grouped(p, [
            (0.0, "fluid", 0, "f0.vtu"),
            (0.0, "solid", 0, "s0.vtu"),
            (1.0, "fluid", 0, "f1.vtu"),
        ])
        text = p.read_text()
        assert 'group="fluid"' in text
        assert 'group="solid"' in text
        assert text.count("<DataSet") == 3
