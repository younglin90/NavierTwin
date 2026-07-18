"""Round 319 — Fluent .cas extension."""

from __future__ import annotations


class TestFluentCasExt:
    def test_section_ids(self) -> None:
        from naviertwin.core.cfd_reader import parse_section_ids

        txt = '(0 "comment") (10 (0 1 2 0 0)) (12 (1 2 3 0 0))'
        ids = parse_section_ids(txt)
        assert 0 in ids
        assert 10 in ids
        assert 12 in ids

    def test_zone_names(self, tmp_path) -> None:
        from naviertwin.core.cfd_reader import list_zone_names

        # synthetic content with two zones
        text = """
        (0 "header")
        (45 (5 inlet velocity-inlet))
        (45 (6 outlet pressure-outlet))
        """
        p = tmp_path / "f.cas"
        p.write_text(text)
        names = list_zone_names(p)
        assert "inlet" in names
        assert "outlet" in names
