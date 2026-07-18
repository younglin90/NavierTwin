"""Round 244 — hashing."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TestHash:
    def test_bytes(self) -> None:
        from naviertwin.utils.hashing import hash_bytes

        assert hash_bytes(b"abc") == hash_bytes(b"abc")
        assert hash_bytes(b"abc") != hash_bytes(b"abd")

    def test_file(self, tmp_path: Path) -> None:
        from naviertwin.utils.hashing import hash_file

        p = tmp_path / "f.bin"
        p.write_bytes(b"\x00\x01\x02")
        h1 = hash_file(p)
        p.write_bytes(b"\x00\x01\x02\x03")
        h2 = hash_file(p)
        assert h1 != h2

    def test_array(self) -> None:
        from naviertwin.utils.hashing import hash_array

        a = np.arange(10, dtype=np.float64)
        b = np.arange(10, dtype=np.float64)
        assert hash_array(a) == hash_array(b)
        # dtype 변경 시 해시 다름
        assert hash_array(a) != hash_array(a.astype(np.float32))

    def test_dict(self) -> None:
        from naviertwin.utils.hashing import hash_dict

        assert hash_dict({"a": 1, "b": 2}) == hash_dict({"b": 2, "a": 1})
        assert hash_dict({"a": 1}) != hash_dict({"a": 2})
