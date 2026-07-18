"""Round 528 — artifact zip."""

from __future__ import annotations

from hashlib import sha256

import pytest


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
        by_name = {m["name"]: m for m in man}
        for fp in [a, b]:
            data = fp.read_bytes()
            entry = by_name[fp.name]
            assert entry["bytes"] == len(data)
            assert entry["sha256"] == sha256(data).hexdigest()

    def test_extra_entries_are_manifested(self, tmp_path) -> None:
        from naviertwin.utils.workflow.artifact_zip import (
            read_manifest,
            zip_artifacts,
        )

        a = tmp_path / "a.txt"
        a.write_text("alpha")
        out = tmp_path / "art.zip"
        zip_artifacts([a], out, extra_entries={"README.txt": "hello"})

        man = read_manifest(out)
        by_name = {m["name"]: m for m in man}
        assert "README.txt" in by_name
        assert by_name["README.txt"]["bytes"] == len(b"hello")
        assert by_name["README.txt"]["sha256"] == sha256(b"hello").hexdigest()

    def test_extra_entries_cannot_overwrite_artifacts(self, tmp_path) -> None:
        from naviertwin.utils.workflow.artifact_zip import zip_artifacts

        a = tmp_path / "a.txt"
        a.write_text("alpha")
        out = tmp_path / "art.zip"

        with pytest.raises(ValueError, match="duplicate or reserved"):
            zip_artifacts([a], out, extra_entries={"a.txt": "replacement"})

    def test_extra_entries_cannot_overwrite_manifest(self, tmp_path) -> None:
        from naviertwin.utils.workflow.artifact_zip import zip_artifacts

        a = tmp_path / "a.txt"
        a.write_text("alpha")
        out = tmp_path / "art.zip"

        with pytest.raises(ValueError, match="duplicate or reserved"):
            zip_artifacts([a], out, extra_entries={"MANIFEST.json": "replacement"})
