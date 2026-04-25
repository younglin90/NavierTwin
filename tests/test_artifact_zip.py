"""Round 528 — artifact zip."""

from __future__ import annotations


class TestArtifactZip:
    def test_round_trip(self, tmp_path) -> None:
        from naviertwin.utils.workflow.artifact_zip import (
            read_manifest,
            zip_artifacts,
        )

        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("alpha")
        b.write_text("beta")
        out = tmp_path / "art.zip"
        zip_artifacts([a, b], out)
        man = read_manifest(out)
        names = sorted(m["name"] for m in man)
        assert names == ["a.txt", "b.txt"]
