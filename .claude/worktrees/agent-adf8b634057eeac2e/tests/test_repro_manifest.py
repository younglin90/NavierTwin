"""Round 567 — repro manifest."""

from __future__ import annotations


class TestRepro:
    def test_keys(self) -> None:
        from naviertwin.utils.repro_manifest import build_manifest

        m = build_manifest(seed=7)
        for k in ["python", "platform", "git_sha", "seed", "packages"]:
            assert k in m
        assert m["seed"] == 7
        assert "numpy" in m["packages"]

    def test_extra_pkgs(self) -> None:
        from naviertwin.utils.repro_manifest import build_manifest

        m = build_manifest(packages=["numpy", "this-pkg-does-not-exist"])
        assert m["packages"]["this-pkg-does-not-exist"] == "missing"
