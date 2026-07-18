"""Round 81 — NumPy-safe JSON."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class TestJsonSafe:
    def test_numpy_types(self) -> None:
        from naviertwin.utils.json_safe import safe_dumps, safe_loads

        d = {
            "i": np.int64(42),
            "f": np.float32(1.5),
            "arr": np.arange(4),
            "b": np.bool_(True),
        }
        s = safe_dumps(d)
        back = safe_loads(s)
        assert back["i"] == 42
        assert abs(back["f"] - 1.5) < 1e-6
        assert back["arr"] == [0, 1, 2, 3]
        assert back["b"] is True

    def test_path_and_dataclass(self) -> None:
        from naviertwin.utils.json_safe import safe_dumps

        @dataclass
        class C:
            x: int
            p: Path

        obj = C(x=3, p=Path("/tmp/x"))
        s = safe_dumps(obj)
        assert '"x": 3' in s
        assert "/tmp/x" in s

    def test_file_roundtrip(self, tmp_path: Path) -> None:
        from naviertwin.utils.json_safe import safe_dump_file, safe_loads

        p = tmp_path / "x.json"
        safe_dump_file({"a": np.array([1.1, 2.2])}, p)
        data = safe_loads(p.read_text())
        assert data["a"] == [1.1, 2.2]

    def test_set_and_bytes(self) -> None:
        from naviertwin.utils.json_safe import safe_dumps, safe_loads

        s = safe_dumps({"s": {1, 2, 3}, "bs": b"hello"})
        d = safe_loads(s)
        assert sorted(d["s"]) == [1, 2, 3]
        assert d["bs"] == "hello"
