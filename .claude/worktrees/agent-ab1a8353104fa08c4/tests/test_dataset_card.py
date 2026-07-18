"""Round 511 — dataset card."""

from __future__ import annotations


class TestCard:
    def test_md_content(self) -> None:
        from naviertwin.utils.dataset_card import dataset_card_md

        md = dataset_card_md(
            name="cavity_re100", n_samples=200, source="DNS",
            schema={"u": "float64 (N, 3)", "p": "float64 (N,)"},
        )
        assert "# Dataset" in md
        assert "cavity_re100" in md
        assert "u" in md

    def test_write(self, tmp_path) -> None:
        from naviertwin.utils.dataset_card import write_card

        p = tmp_path / "card.md"
        write_card(p, name="x", n_samples=10)
        assert p.exists()
        assert "Samples" in p.read_text()
