"""Round 205 — pipeline builder."""

from __future__ import annotations


class TestBuilder:
    def test_defaults(self) -> None:
        from naviertwin.core.digital_twin.pipeline_builder import build_pipeline

        p = build_pipeline({})
        assert p.reducer_kind == "pod"
        assert p.n_modes == 5

    def test_custom(self) -> None:
        from naviertwin.core.digital_twin.pipeline_builder import build_pipeline

        p = build_pipeline({"reducer_kind": "pod", "n_modes": 7, "surrogate_kind": "rbf"})
        assert p.n_modes == 7
        assert p.surrogate_kind == "rbf"

    def test_validate_ok(self) -> None:
        from naviertwin.core.digital_twin.pipeline_builder import validate_config

        assert validate_config({"reducer_kind": "pod", "n_modes": 3}) == []
        assert validate_config({"reducer_kind": "incremental_pod", "n_modes": 3}) == []
        assert validate_config({"reducer_kind": "mrpod", "n_modes": 3}) == []

    def test_validate_issues(self) -> None:
        from naviertwin.core.digital_twin.pipeline_builder import validate_config

        issues = validate_config({
            "reducer_kind": "bogus",
            "surrogate_kind": "foo",
            "n_modes": -1,
        })
        assert len(issues) == 3
