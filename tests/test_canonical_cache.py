"""Canonical `.ntwin` 캐시 테스트 — 저장 계층 단계 2 (외부 검토 §6½ #6).

지키려는 계약:
    - (a) 첫 로드는 miss → ``reader_fn`` 호출 1회. 두 번째 로드(같은 원본,
      mtime/size 불변)는 hit → ``reader_fn`` 호출 0회(캐시에서 바로 복원).
      복원된 데이터셋의 필드/좌표는 원본과 값 수준(bit) 동일하다.
    - (b) 원본 파일이 바뀌면(mtime 변경) 캐시가 자동으로 무효화된다 —
      ``reader_fn`` 이 다시 호출된다.
    - (c) ``use_canonical_cache=False``(기본값)는 매번 ``reader_fn`` 을
      호출하는 기존 동작과 100% 동일하다 (하위 호환 회귀 고정).

실제 파일 I/O 는 ``tmp_path`` 로 격리한다. demo 데이터 대신 작은 합성
UnstructuredGrid 를 ``.vtk`` 로 저장해 ``ReaderFactory`` 로 읽는 흐름을
그대로 재현한다. 좌표/필드 값은 float32 로도 정확히 표현되는 작은 정수만
써서(0, 1, 2 ...), `.ntwin` 캐시가 내부적으로 float32 로 저장하더라도
원본과 값이 정확히(bit) 일치하도록 한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="canonical 캐시 테스트에는 pyvista 가 필요합니다.")
pytest.importorskip("h5py", reason="canonical 캐시는 .ntwin(h5py) 이 필요합니다.")

from naviertwin.core.storage.canonical_cache import CanonicalCache  # noqa: E402
from naviertwin.web import service  # noqa: E402


def _make_tetra_vtk(path: Path) -> None:
    """정수 좌표/필드값을 가진 사면체 1개짜리 합성 메쉬를 .vtk 로 저장한다."""
    import pyvista as pv

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cells = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    celltypes = np.array([pv.CellType.TETRA], dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells, celltypes, points)
    mesh.point_data["p"] = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    mesh.save(str(path))


# ──────────────────────────────────────────────────────────────────────
# (a) miss → reader_fn 호출, hit → reader_fn 호출 안 함 + 값 bit-동일
# ──────────────────────────────────────────────────────────────────────


def test_get_or_convert_hit_skips_reader_and_is_value_identical(tmp_path: Path) -> None:
    src = tmp_path / "case.vtk"
    _make_tetra_vtk(src)

    calls = {"n": 0}

    def counting_reader(p: Path):
        calls["n"] += 1
        from naviertwin.core.cfd_reader import ReaderFactory

        return ReaderFactory().create_and_read(p)

    cache = CanonicalCache(cache_dir=tmp_path / "cache")

    first = cache.get_or_convert(src, counting_reader)
    assert calls["n"] == 1  # miss → reader_fn 호출

    second = cache.get_or_convert(src, counting_reader)
    assert calls["n"] == 1  # hit → reader_fn 재호출 없음

    # 필드/좌표가 값 수준으로 bit-동일 (dtype 은 캐시 왕복 중 float32 로
    # 바뀔 수 있으므로 float64 로 캐스팅해 비교한다 — 원본 값이 정수라
    # float32 변환에서도 정밀도 손실이 없다).
    p1 = np.asarray(first.mesh.points, dtype=np.float64)
    p2 = np.asarray(second.mesh.points, dtype=np.float64)
    assert np.array_equal(p1, p2)

    f1 = np.asarray(first.mesh.point_data["p"], dtype=np.float64)
    f2 = np.asarray(second.mesh.point_data["p"], dtype=np.float64)
    assert np.array_equal(f1, f2)

    assert first.n_points == second.n_points

    # 캐시 항목이 실제로 디스크에 하나 생겼다.
    assert cache.stats()["n_entries"] == 1


# ──────────────────────────────────────────────────────────────────────
# (b) mtime 변경 → 캐시 무효화(재호출)
# ──────────────────────────────────────────────────────────────────────


def test_mtime_change_invalidates_cache(tmp_path: Path) -> None:
    import os
    import time

    src = tmp_path / "case.vtk"
    _make_tetra_vtk(src)

    calls = {"n": 0}

    def counting_reader(p: Path):
        calls["n"] += 1
        from naviertwin.core.cfd_reader import ReaderFactory

        return ReaderFactory().create_and_read(p)

    cache = CanonicalCache(cache_dir=tmp_path / "cache")
    cache.get_or_convert(src, counting_reader)
    assert calls["n"] == 1

    cache.get_or_convert(src, counting_reader)
    assert calls["n"] == 1  # 변경 전에는 hit

    # 원본을 다시 저장해 mtime 을 갱신한다(내용이 같아도 무효화되어야 함).
    time.sleep(0.01)
    _make_tetra_vtk(src)
    new_stat = src.stat()
    os.utime(src, ns=(new_stat.st_atime_ns, new_stat.st_mtime_ns + 1_000_000_000))

    cache.get_or_convert(src, counting_reader)
    assert calls["n"] == 2  # mtime 이 바뀌었으니 재호출(miss)

    cache.get_or_convert(src, counting_reader)
    assert calls["n"] == 2  # 새 mtime 으로 다시 hit


# ──────────────────────────────────────────────────────────────────────
# (c) use_canonical_cache=False(기본) → 기존 동작과 100% 동일
# ──────────────────────────────────────────────────────────────────────


def test_load_dataset_default_bypasses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "case.vtk"
    _make_tetra_vtk(src)

    import naviertwin.core.cfd_reader as cfd_reader_pkg

    calls = {"n": 0}
    real_create_and_read = cfd_reader_pkg.ReaderFactory.create_and_read

    def counting_create_and_read(self, path):
        calls["n"] += 1
        return real_create_and_read(path)

    monkeypatch.setattr(
        cfd_reader_pkg.ReaderFactory, "create_and_read", counting_create_and_read
    )

    # 기본값(use_canonical_cache=False) — 매번 reader_fn(ReaderFactory) 이
    # 직접 호출된다. 캐시를 전혀 만들지 않는다.
    service.load_dataset(src)
    assert calls["n"] == 1
    service.load_dataset(src)
    assert calls["n"] == 2
    service.load_dataset(src)
    assert calls["n"] == 3


def test_load_dataset_use_canonical_cache_true_hits_after_first_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "case.vtk"
    _make_tetra_vtk(src)

    import naviertwin.core.cfd_reader as cfd_reader_pkg

    calls = {"n": 0}
    real_create_and_read = cfd_reader_pkg.ReaderFactory.create_and_read

    def counting_create_and_read(self, path):
        calls["n"] += 1
        return real_create_and_read(path)

    monkeypatch.setattr(
        cfd_reader_pkg.ReaderFactory, "create_and_read", counting_create_and_read
    )

    cache_dir = tmp_path / "cache"
    ds1 = service.load_dataset(src, use_canonical_cache=True, canonical_cache_dir=cache_dir)
    assert calls["n"] == 1
    ds2 = service.load_dataset(src, use_canonical_cache=True, canonical_cache_dir=cache_dir)
    assert calls["n"] == 1  # hit — ReaderFactory 재호출 없음

    assert ds1.n_points == ds2.n_points
    assert CanonicalCache(cache_dir=cache_dir).stats()["n_entries"] == 1
