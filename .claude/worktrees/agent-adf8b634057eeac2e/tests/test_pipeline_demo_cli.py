"""Customer-facing pipeline demo CLI tests."""

from __future__ import annotations

import json

import pytest


def test_pipeline_demo_parser_requires_outdir(tmp_path) -> None:
    from naviertwin.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["pipeline-demo", "--outdir", str(tmp_path)])

    assert args.command == "pipeline-demo"
    assert args.outdir == str(tmp_path)
    assert args.n_modes == 3
    assert args.surrogate == "rbf"

    with pytest.raises(SystemExit):
        parser.parse_args(["pipeline-demo"])


def test_run_pipeline_demo_writes_customer_artifacts(tmp_path, capsys) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("jinja2")
    from naviertwin.main import _run_pipeline_demo

    code = _run_pipeline_demo(outdir=str(tmp_path), n_modes=3, surrogate="rbf")
    stdout_payload = json.loads(capsys.readouterr().out)
    metrics_path = tmp_path / "metrics.json"
    report_path = tmp_path / "report.html"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert code == 0
    assert stdout_payload["status"] == "ok"
    assert metrics_payload["status"] == "ok"
    assert metrics_payload["artifacts"]["report"] == str(report_path)
    assert "rmse" in metrics_payload["metrics"]
    assert metrics_payload["metrics"]["r2"] >= 0.9
    assert report_path.exists()
    assert "NavierTwin Pipeline Demo" in report_path.read_text(encoding="utf-8")


def test_run_pipeline_demo_reports_missing_dependency_without_traceback(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    import builtins

    from naviertwin.main import _run_pipeline_demo

    real_import = builtins.__import__

    def block_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("blocked numpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_numpy)

    code = _run_pipeline_demo(outdir=str(tmp_path), n_modes=3, surrogate="rbf")
    output = capsys.readouterr()

    assert code == 2
    assert output.out == ""
    assert "pipeline-demo error:" in output.err
    assert "Traceback" not in output.err
