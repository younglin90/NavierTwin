"""Round 83 — CSV → snapshots 로더."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_csv(path: Path, data: dict) -> None:
    import pandas as pd

    pd.DataFrame(data).to_csv(path, index=False)


class TestCsvSnapshots:
    def test_basic(self, tmp_path: Path) -> None:
        pytest.importorskip("pandas")
        from naviertwin.core.cfd_reader.csv_snapshots import load_csv_snapshots

        paths = []
        n = 8
        for i in range(4):
            p = tmp_path / f"t{i:03d}.csv"
            _write_csv(p, {
                "x": np.arange(n), "y": np.zeros(n), "z": np.zeros(n),
                "U": np.arange(n) * (i + 1),
            })
            paths.append(p)

        X, coords = load_csv_snapshots(paths, column="U")
        assert X.shape == (n, 4)
        assert coords is not None and coords.shape == (n, 3)
        # 두 번째 스냅샷 = 2*첫 스냅샷(+1 offset)
        assert np.allclose(X[:, 1], np.arange(n) * 2)

    def test_missing_column(self, tmp_path: Path) -> None:
        pytest.importorskip("pandas")
        from naviertwin.core.cfd_reader.csv_snapshots import load_csv_snapshots

        p = tmp_path / "a.csv"
        _write_csv(p, {"x": [1, 2], "V": [3, 4]})
        with pytest.raises(KeyError):
            load_csv_snapshots([p], column="U")

    def test_size_mismatch(self, tmp_path: Path) -> None:
        pytest.importorskip("pandas")
        from naviertwin.core.cfd_reader.csv_snapshots import load_csv_snapshots

        _write_csv(tmp_path / "a.csv", {"U": [1, 2, 3]})
        _write_csv(tmp_path / "b.csv", {"U": [1, 2]})
        with pytest.raises(ValueError):
            load_csv_snapshots(
                [tmp_path / "a.csv", tmp_path / "b.csv"], column="U",
            )
