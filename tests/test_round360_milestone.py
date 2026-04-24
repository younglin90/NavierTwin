"""Round 360 — FINAL milestone J: serving/ops + schema + full project e2e."""

from __future__ import annotations


class TestMilestoneJ:
    def test_imports(self) -> None:
        from naviertwin.core.serving import grpc_server  # noqa: F401
        from naviertwin.utils import (  # noqa: F401
            cli_entry,
            docker_gen,
            github_ci_gen,
            otel,
            precommit_gen,
            prom_exporter,
            rate_limit_http,
            schema_validator,
            sphinx_autoapi_gen,
        )

    def test_schema_validator(self) -> None:
        from naviertwin.utils.schema_validator import validate

        data = {"epochs": 10, "lr": 0.01, "name": "exp1"}
        schema = {"epochs": int, "lr": lambda x: 0 < x < 1, "name": str}
        ok, errs = validate(data, schema)
        assert ok
        assert errs == []
        # bad data
        ok, errs = validate({"epochs": "10", "lr": 5.0, "name": 1}, schema)
        assert not ok
        assert len(errs) >= 3

    def test_full_e2e(self, tmp_path) -> None:
        """Full pipeline: write Dockerfile, CI yaml, schema-validate config."""
        from naviertwin.utils.docker_gen import write_dockerfile
        from naviertwin.utils.github_ci_gen import write_ci
        from naviertwin.utils.precommit_gen import write_precommit
        from naviertwin.utils.prom_exporter import MetricsRegistry
        from naviertwin.utils.schema_validator import validate

        write_dockerfile(tmp_path / "Dockerfile")
        write_ci(tmp_path / ".github/workflows/ci.yml")
        write_precommit(tmp_path / ".pre-commit-config.yaml")
        for f in ["Dockerfile", ".github/workflows/ci.yml", ".pre-commit-config.yaml"]:
            assert (tmp_path / f).exists()
        # metrics
        reg = MetricsRegistry()
        reg.counter("e2e").inc()
        assert "e2e 1" in reg.format()
        # schema
        ok, _ = validate({"x": 1}, {"x": int})
        assert ok

    def test_cli_help_runs(self) -> None:
        """CLI parser builds without error."""
        from naviertwin.utils.cli_entry import build_parser

        p = build_parser()
        assert p.prog == "naviertwin"
