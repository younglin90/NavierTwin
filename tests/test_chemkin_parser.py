"""Round 487 — CHEMKIN parser."""

from __future__ import annotations


class TestChemkin:
    def test_basic(self) -> None:
        from naviertwin.core.reaction.chemkin_parser import parse_reaction

        r = parse_reaction("H2 + O = OH + H  1.04e8 0.0 6.96")
        assert r["reactants"] == ["H2", "O"]
        assert r["products"] == ["OH", "H"]
        assert abs(r["A"] - 1.04e8) < 1e-3
        assert r["beta"] == 0.0
        assert r["Ea"] == 6.96

    def test_no_rates(self) -> None:
        from naviertwin.core.reaction.chemkin_parser import parse_reaction

        r = parse_reaction("A = B")
        assert r["reactants"] == ["A"]
        assert r["products"] == ["B"]
        assert r["A"] == 0.0
