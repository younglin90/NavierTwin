"""Round 355 — CLI entry."""

from __future__ import annotations


class TestCLI:
    def test_train(self) -> None:
        from naviertwin.utils.cli_entry import build_parser

        p = build_parser()
        ns = p.parse_args(["train", "--epochs", "5", "--config", "x.toml"])
        assert ns.cmd == "train"
        assert ns.epochs == 5
        assert ns.config == "x.toml"

    def test_predict(self) -> None:
        from naviertwin.utils.cli_entry import build_parser

        p = build_parser()
        ns = p.parse_args(["predict", "--input", "x.h5"])
        assert ns.cmd == "predict"
        assert ns.input == "x.h5"

    def test_main(self, capsys) -> None:
        from naviertwin.utils.cli_entry import main

        rc = main(["info"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "info" in out
