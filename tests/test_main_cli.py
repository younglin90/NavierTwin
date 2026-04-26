"""Round 592 — main.py CLI path coverage."""

from __future__ import annotations

import sys

import pytest


class TestBuildParser:
    def test_parser_created(self) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        assert p.prog == "naviertwin"

    def test_parse_no_args(self) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args([])
        assert args.gui is False
        assert args.command is None

    def test_parse_gui_flag(self) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(["--gui"])
        assert args.gui is True

    def test_parse_benchmark_subcommand(self) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(["benchmark", "--kind", "burgers"])
        assert args.command == "benchmark"
        assert args.kind == "burgers"

    def test_parse_server_subcommand(self) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(["server", "--port", "9000"])
        assert args.command == "server"
        assert args.port == 9000

    def test_parse_pipeline_subcommand(self) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(["pipeline", "--reducer", "ae", "--n-modes", "8"])
        assert args.command == "pipeline"
        assert args.reducer == "ae"
        assert args.n_modes == 8


class TestRunGui:
    def test_run_gui_no_pyside6(self, monkeypatch) -> None:
        import builtins

        from naviertwin.main import _run_gui

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "PySide6.QtWidgets":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        code = _run_gui(None)
        assert code == 1


class TestRunBenchmark:
    def test_run_benchmark_unknown_kind(self) -> None:
        from naviertwin.main import _run_benchmark

        code = _run_benchmark("unknown_kind")
        assert code == 1


class TestRunServer:
    def test_run_server_no_uvicorn(self, monkeypatch) -> None:
        import builtins

        from naviertwin.main import _run_server

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "uvicorn":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        code = _run_server("127.0.0.1", 8000)
        assert code == 1


class TestRunPipeline:
    def test_run_pipeline_pod_kriging(self) -> None:
        from naviertwin.main import _run_pipeline

        code = _run_pipeline("pod", 3, "kriging")
        assert code == 0

    def test_run_pipeline_ae_rbf(self) -> None:
        from naviertwin.main import _run_pipeline

        code = _run_pipeline("ae", 2, "rbf")
        assert code == 0


class TestMain:
    def test_main_no_args_exits_zero(self, monkeypatch) -> None:
        from naviertwin.main import main

        monkeypatch.setattr(sys, "argv", ["naviertwin"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
