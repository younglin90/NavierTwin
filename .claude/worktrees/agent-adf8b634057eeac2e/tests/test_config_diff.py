"""Round 243 — config diff."""

from __future__ import annotations


class TestDiff:
    def test_basic(self) -> None:
        from naviertwin.utils.config_diff import diff_configs

        a = {"x": 1, "y": 2}
        b = {"x": 1, "y": 3, "z": 4}
        d = diff_configs(a, b)
        assert d["changed"] == {"y": (2, 3)}
        assert d["added"] == {"z": 4}
        assert d["removed"] == {}

    def test_nested(self) -> None:
        from naviertwin.utils.config_diff import diff_configs

        a = {"m": {"lr": 1e-3, "layers": [1, 2]}}
        b = {"m": {"lr": 5e-4, "layers": [1, 2], "drop": 0.1}}
        d = diff_configs(a, b)
        assert "m.lr" in d["changed"]
        assert "m.drop" in d["added"]

    def test_format(self) -> None:
        from naviertwin.utils.config_diff import diff_configs, format_diff

        a = {"x": 1}
        b = {"x": 2}
        s = format_diff(diff_configs(a, b))
        assert "~ x" in s
