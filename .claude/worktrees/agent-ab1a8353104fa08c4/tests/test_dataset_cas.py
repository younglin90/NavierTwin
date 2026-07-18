"""Round 514 — dataset CAS."""

from __future__ import annotations


class TestCAS:
    def test_distinct(self) -> None:
        from naviertwin.utils.dataset_cas import cas_hash

        assert cas_hash(b"a") != cas_hash(b"b")

    def test_file(self, tmp_path) -> None:
        from naviertwin.utils.dataset_cas import cas_hash, cas_hash_file

        p = tmp_path / "f.bin"
        p.write_bytes(b"hello world")
        assert cas_hash_file(p) == cas_hash(b"hello world")

    def test_version_id(self) -> None:
        from naviertwin.utils.dataset_cas import cas_hash, version_id

        h = cas_hash(b"x")
        v = version_id("ds", h)
        assert v.startswith("ds@")
